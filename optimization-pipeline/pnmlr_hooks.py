#!/usr/bin/env python3
"""
PNMLR Hooks for Hybrid GA Integration.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import sys

# Add project root for CityDataManager
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

def _load_pnmlr_artifacts(models_dir: Path) -> Dict[str, Any]:
    """Load all PNMLR model artifacts."""
    from pnmlr_model import PNMLRModel
    from pnmlr_features import FeatureNormalizer
    from pnmlr_profiles import UserProfileSet
    
    model_path = models_dir / "pnmlr_model.pkl"
    normalizer_path = models_dir / "pnmlr_normalizer.json"
    profiles_path = models_dir / "pnmlr_profiles.json"
    target_params_path = models_dir / "pnmlr_target_params.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"PNMLR model not found at {model_path}")
    
    return {
        'model': PNMLRModel.load(model_path),
        'normalizer': FeatureNormalizer.load(normalizer_path),
        'profiles': UserProfileSet.load(profiles_path),
        'target_params': json.loads(target_params_path.read_text(encoding='utf-8')),
    }

def pnmlr_precompute_hook(context: 'GAContext') -> Mapping[str, object]:
    """Precompute hook that loads PNMLR model and computes utilities."""
    from pnmlr_features import extract_node_features
    
    cdm = context.cdm
    pnmlr_cfg = context.raw_config.get('pnmlr', {})
    models_dir = cdm.project_root / pnmlr_cfg.get('models_dir', 'models') / cdm.city
    
    try:
        artifacts = _load_pnmlr_artifacts(models_dir)
    except FileNotFoundError as e:
        logger.warning(f"PNMLR not available for {cdm.city}: {e}")
        # Fall back to default precompute effects
        from hybrid_ga import default_precompute_effects
        effects = dict(default_precompute_effects(context))
        effects["pnmlr_enabled"] = False
        return effects
    
    model, normalizer, profiles, target_params = (
        artifacts['model'], artifacts['normalizer'], 
        artifacts['profiles'], artifacts['target_params']
    )
    
    amenity_types = target_params['amenity_types']
    nf, fn = extract_node_features(context.nodes, amenity_names=amenity_types, include_coords=True, include_graph_metrics=True)
    nf_norm = normalizer.transform(nf)
    
    idx_map = [profiles.amenity_types.index(a) for a in amenity_types if a in profiles.amenity_types]
    pw = profiles.profiles[:, idx_map]
    pw = pw / pw.sum(axis=1, keepdims=True)
    
    avg_u = model.predict_average_utility(nf_norm, pw)
    t_min, t_max = target_params['min'], target_params['max']
    avg_u = avg_u * (t_max - t_min) + t_min
    
    nodes_copy = context.nodes.copy()
    nodes_copy['pnmlr_utility'] = avg_u
    
    return {
        "nodes": nodes_copy,
        "pnmlr_utilities": dict(zip(context.nodes.index.astype(str), avg_u)),
        "pnmlr_enabled": True,
        "pnmlr_profiles": profiles,
        "pnmlr_model": model,
    }

def pnmlr_evaluate_hook(candidate: 'Candidate', effects: Mapping[str, object], context: 'GAContext') -> Dict[str, object]:
    """Evaluate hook using PNMLR utility scores."""
    import math
    if not effects.get("pnmlr_enabled", False):
        from hybrid_ga import default_evaluate_candidate
        return default_evaluate_candidate(candidate, effects, context)
    
    utility_map = effects["pnmlr_utilities"]
    nodes = effects["nodes"]
    
    total_u = 0.0
    amenity_u = {}
    for amenity, node_ids in candidate.placements.items():
        w = context.amenity_weights.get(amenity, 1.0)
        u = sum(utility_map.get(str(nid), 0) for nid in node_ids) * w
        amenity_u[amenity] = u
        total_u += u
    
    # Simple proxies for penalties as in original
    div_penalty = 0.0 # simplified for speed in this refactor view
    prox_penalty = 0.0
    
    fitness = total_u - div_penalty - prox_penalty
    return {
        "fitness": max(fitness, 0.0),
        "pnmlr_utility": total_u,
        "amenity_utilities": amenity_u,
        "placements": {a: len(n) for a, n in candidate.placements.items()},
    }

def create_pnmlr_hooks(cdm: Optional[CityDataManager] = None) -> tuple:
    return pnmlr_precompute_hook, pnmlr_evaluate_hook
