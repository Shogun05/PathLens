#!/usr/bin/env python3
"""
PNMLR Hooks for Hybrid GA Integration.

Provides precompute and evaluate hooks that integrate the PNMLR model
into the PathLens genetic algorithm optimization pipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

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
        raise FileNotFoundError(f"PNMLR model not found at {model_path}. Run train_pnmlr.py first.")
    
    model = PNMLRModel.load(model_path)
    normalizer = FeatureNormalizer.load(normalizer_path)
    profiles = UserProfileSet.load(profiles_path)
    
    target_params = json.loads(target_params_path.read_text(encoding='utf-8'))
    
    logger.info(f"Loaded PNMLR model from {models_dir}")
    logger.info(f"  - {profiles.n_profiles} user profiles")
    logger.info(f"  - {len(target_params['amenity_types'])} amenity types")
    
    return {
        'model': model,
        'normalizer': normalizer,
        'profiles': profiles,
        'target_params': target_params,
    }


def pnmlr_precompute_hook(context: 'GAContext') -> Mapping[str, object]:
    """
    Precompute hook that loads PNMLR model and pre-computes utilities.
    
    This hook:
    1. Loads the trained PNMLR model and profiles
    2. Extracts features for all nodes
    3. Pre-computes average utility scores across all profiles
    4. Stores the node→utility mapping for fast lookup during evaluation
    
    Args:
        context: GAContext with nodes DataFrame and configuration
    
    Returns:
        Effects dict with 'nodes' (updated) and 'pnmlr_utilities'
    """
    from pnmlr_features import extract_node_features
    
    # Determine models directory
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    
    # Load artifacts
    try:
        artifacts = _load_pnmlr_artifacts(models_dir)
    except FileNotFoundError as e:
        logger.warning(f"PNMLR not available: {e}. Falling back to default evaluation.")
        return {"nodes": context.nodes.copy(), "pnmlr_enabled": False}
    
    model = artifacts['model']
    normalizer = artifacts['normalizer']
    profiles = artifacts['profiles']
    target_params = artifacts['target_params']
    
    # Extract features for all nodes
    amenity_types = target_params['amenity_types']
    nodes_df = context.nodes
    
    logger.info(f"Running PNMLR precompute for {len(nodes_df)} nodes.")
    
    node_features, feature_names = extract_node_features(
        nodes_df,
        amenity_names=amenity_types,
        include_coords=True,
        include_graph_metrics=True,
    )
    
    # Normalize features using saved params
    node_features_norm = normalizer.transform(node_features)
    
    # Get profile weights for available amenities
    profile_amenity_indices = [
        profiles.amenity_types.index(a) 
        for a in amenity_types 
        if a in profiles.amenity_types
    ]
    profile_weights = profiles.profiles[:, profile_amenity_indices]
    # Renormalize
    profile_weights = profile_weights / profile_weights.sum(axis=1, keepdims=True)
    
    # Predict average utility across all profiles
    avg_utilities = model.predict_average_utility(node_features_norm, profile_weights)
    
    # Denormalize utilities back to original scale
    target_min = target_params['min']
    target_max = target_params['max']
    avg_utilities = avg_utilities * (target_max - target_min) + target_min
    
    logger.info(
        f"PNMLR utilities computed: range=[{avg_utilities.min():.4f}, {avg_utilities.max():.4f}], "
        f"mean={avg_utilities.mean():.4f}"
    )
    
    # Create node_id → utility mapping
    node_ids = nodes_df.index.astype(str).tolist()
    utility_map = dict(zip(node_ids, avg_utilities))
    
    # Add utilities to nodes DataFrame
    nodes_copy = nodes_df.copy()
    nodes_copy['pnmlr_utility'] = avg_utilities
    
    return {
        "nodes": nodes_copy,
        "pnmlr_utilities": utility_map,
        "pnmlr_enabled": True,
        "pnmlr_profiles": profiles,
        "pnmlr_model": model,
    }


def pnmlr_evaluate_hook(
    candidate: 'Candidate',
    effects: Mapping[str, object],
    context: 'GAContext',
) -> Dict[str, object]:
    """
    Evaluate hook using PNMLR utility scores.
    
    Fitness is computed as the average predicted utility across all user
    profiles for the placed amenities, penalized for diversity and proximity.
    
    This enables amenity placement decisions that improve accessibility
    for diverse population groups rather than a single average user.
    
    Args:
        candidate: Candidate with placements dict
        effects: Precomputed effects from pnmlr_precompute_hook
        context: GAContext
    
    Returns:
        Metrics dict with fitness, utility scores, etc.
    """
    import math
    
    nodes: pd.DataFrame = effects["nodes"]  # type: ignore
    pnmlr_enabled = effects.get("pnmlr_enabled", False)
    utility_map = effects.get("pnmlr_utilities", {})
    
    if not pnmlr_enabled:
        # Fall back to default evaluation
        from hybrid_ga import default_evaluate_candidate
        return default_evaluate_candidate(candidate, effects, context)
    
    amenity_weights = context.amenity_weights
    
    # Compute total utility gain from placements
    total_utility = 0.0
    placements: Dict[str, int] = {}
    amenity_utilities: Dict[str, float] = {}
    
    for amenity, node_ids in candidate.placements.items():
        weight = amenity_weights.get(amenity, 1.0)
        amenity_utility = 0.0
        
        for node_id in node_ids:
            node_str = str(node_id)
            if node_str in utility_map:
                # Weight utility by amenity importance
                amenity_utility += utility_map[node_str] * weight
        
        amenity_utilities[amenity] = amenity_utility
        total_utility += amenity_utility
        placements[amenity] = len(node_ids)
    
    # Diversity penalty: penalize same-type amenities placed too close
    diversity_penalty = 0.0
    for amenity, node_ids in candidate.placements.items():
        if len(node_ids) <= 1:
            continue
        
        coords = []
        for node_id in node_ids:
            node_str = str(node_id)
            if node_str in nodes.index and 'x' in nodes.columns and 'y' in nodes.columns:
                x = nodes.loc[node_str, 'x']
                y = nodes.loc[node_str, 'y']
                if pd.notna(x) and pd.notna(y):
                    coords.append((float(x), float(y)))
        
        # Pairwise distance check
        min_spacing = 800.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                if dist < min_spacing:
                    ratio = dist / min_spacing
                    diversity_penalty += (1.0 - ratio) ** 2 * 2.0
    
    # Proximity penalty: penalize placing too close to existing amenities
    proximity_penalty = 0.0
    for amenity, node_ids in candidate.placements.items():
        col = context.distance_columns.get(amenity)
        if not col or col not in nodes.columns:
            continue
        
        for node_id in node_ids:
            node_str = str(node_id)
            if node_str in nodes.index:
                existing_dist = pd.to_numeric(nodes.loc[node_str, col], errors='coerce')
                if pd.notna(existing_dist) and existing_dist < 600:
                    ratio = existing_dist / 600.0
                    proximity_penalty += (1.0 - ratio) ** 2 * 0.5
    
    # Travel time penalty (mild)
    travel_penalty = 0.0
    if 'travel_time_min' in nodes.columns:
        placed_indices = [str(node) for ids in candidate.placements.values() for node in ids]
        placed_nodes = nodes.loc[nodes.index.isin(placed_indices)]
        if not placed_nodes.empty:
            travel_penalty = float(placed_nodes['travel_time_min'].mean())
    
    # Final fitness: utility - penalties
    fitness = total_utility - diversity_penalty - proximity_penalty - 0.0005 * travel_penalty
    fitness = max(fitness, 0.0)
    
    # Compute best_distances for backward compatibility
    best_distances: Dict[str, float] = {}
    for amenity, node_ids in candidate.placements.items():
        col = context.distance_columns.get(amenity)
        if col and col in nodes.columns:
            target_nodes = [str(node) for node in node_ids]
            distances = nodes.loc[nodes.index.isin(target_nodes), col]
            numeric_distances = pd.to_numeric(distances, errors="coerce").dropna()
            if not numeric_distances.empty:
                best_distances[amenity] = float(numeric_distances.min())
    
    # Return with backward-compatible field names
    return {
        "fitness": fitness,
        # Original field names for backward compatibility
        "distance_gain": total_utility,  # PNMLR utility replaces distance_gain
        "travel_penalty": travel_penalty,
        "diversity_penalty": diversity_penalty,
        "proximity_penalty": proximity_penalty,
        "placements": placements,
        "best_distances": best_distances,
        "amenity_scores": amenity_utilities,  # PNMLR utilities per amenity
        # Additional PNMLR-specific fields
        "pnmlr_utility": total_utility,
        "amenity_utilities": amenity_utilities,
    }


def create_pnmlr_hooks(models_dir: Optional[Path] = None) -> tuple:
    """
    Factory function to create PNMLR hooks with custom models directory.
    
    Args:
        models_dir: Optional path to models directory
    
    Returns:
        Tuple of (precompute_hook, evaluate_hook)
    """
    if models_dir is not None:
        # Create closure with custom path
        def precompute(context):
            from pnmlr_features import extract_node_features
            
            try:
                artifacts = _load_pnmlr_artifacts(models_dir)
            except FileNotFoundError as e:
                logger.warning(f"PNMLR not available: {e}")
                return {"nodes": context.nodes.copy(), "pnmlr_enabled": False}
            
            # (rest of precompute logic...)
            model = artifacts['model']
            normalizer = artifacts['normalizer']
            profiles = artifacts['profiles']
            target_params = artifacts['target_params']
            
            amenity_types = target_params['amenity_types']
            nodes_df = context.nodes
            
            node_features, _ = extract_node_features(
                nodes_df,
                amenity_names=amenity_types,
                include_coords=True,
                include_graph_metrics=True,
            )
            
            node_features_norm = normalizer.transform(node_features)
            
            profile_amenity_indices = [
                profiles.amenity_types.index(a) 
                for a in amenity_types 
                if a in profiles.amenity_types
            ]
            profile_weights = profiles.profiles[:, profile_amenity_indices]
            profile_weights = profile_weights / profile_weights.sum(axis=1, keepdims=True)
            
            avg_utilities = model.predict_average_utility(node_features_norm, profile_weights)
            
            target_min = target_params['min']
            target_max = target_params['max']
            avg_utilities = avg_utilities * (target_max - target_min) + target_min
            
            node_ids = nodes_df.index.astype(str).tolist()
            utility_map = dict(zip(node_ids, avg_utilities))
            
            nodes_copy = nodes_df.copy()
            nodes_copy['pnmlr_utility'] = avg_utilities
            
            return {
                "nodes": nodes_copy,
                "pnmlr_utilities": utility_map,
                "pnmlr_enabled": True,
            }
        
        return precompute, pnmlr_evaluate_hook
    
    return pnmlr_precompute_hook, pnmlr_evaluate_hook


if __name__ == "__main__":
    # Test hook loading
    logging.basicConfig(level=logging.INFO)
    
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    
    print(f"Testing PNMLR hooks with models from {models_dir}")
    
    if (models_dir / "pnmlr_model.pkl").exists():
        artifacts = _load_pnmlr_artifacts(models_dir)
        print("Successfully loaded PNMLR artifacts:")
        print(f"  - Model: {artifacts['model']}")
        print(f"  - Profiles: {artifacts['profiles'].n_profiles}")
    else:
        print("No trained model found. Run train_pnmlr.py first.")
