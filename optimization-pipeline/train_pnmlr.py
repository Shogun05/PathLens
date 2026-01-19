#!/usr/bin/env python3
"""
PNMLR Training Script.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root for CityDataManager
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pnmlr_features import (
    extract_node_features,
    get_amenity_names,
    get_amenity_distance_columns,
    compute_profile_utility,
    FeatureNormalizer,
)
from pnmlr_profiles import UserProfileSet
from pnmlr_model import PNMLRModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def generate_training_data(nodes_df: pd.DataFrame, profiles: UserProfileSet, decay_constant: float = 2000.0):
    amenity_names = profiles.amenity_types
    distance_cols = [f"dist_to_{a}" for a in amenity_names if f"dist_to_{a}" in nodes_df.columns]
    available_amenities = [col.replace("dist_to_", "") for col in distance_cols]
    
    if len(available_amenities) < len(amenity_names):
        logger.warning(f"Using {len(available_amenities)} amenities: {available_amenities}")
        amenity_indices = [profiles.amenity_types.index(a) for a in available_amenities]
        profile_weights = profiles.profiles[:, amenity_indices]
        profile_weights = profile_weights / profile_weights.sum(axis=1, keepdims=True)
    else:
        profile_weights = profiles.profiles
        available_amenities = amenity_names
    
    distances = np.column_stack([pd.to_numeric(nodes_df[f"dist_to_{a}"], errors='coerce').fillna(np.inf).values for a in available_amenities])
    N, P = len(nodes_df), profiles.n_profiles
    
    all_utilities = [compute_profile_utility(distances, profile_weights[p], decay_constant) for p in range(P)]
    targets = np.column_stack(all_utilities).flatten()
    
    node_features, feature_names = extract_node_features(nodes_df, amenity_names=available_amenities, include_coords=True, include_graph_metrics=True)
    node_features_repeated = np.tile(node_features, (P, 1))
    profile_weights_tiled = np.repeat(profile_weights, N, axis=0)
    
    return node_features_repeated, profile_weights_tiled, targets, feature_names, available_amenities

def train_pnmlr(cdm: CityDataManager, config: dict):
    pnmlr_cfg = config.get('pnmlr', {})
    train_cfg = pnmlr_cfg.get('training', {})
    
    output_dir = cdm.project_root / pnmlr_cfg.get('models_dir', 'models') / cdm.city
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nodes_df = pd.read_parquet(cdm.baseline_nodes)
    logger.info(f"Loaded {len(nodes_df)} nodes for {cdm.city}")
    
    profiles = UserProfileSet.generate(
        n_profiles=train_cfg.get('n_profiles', 20),
        amenity_types=get_amenity_names(nodes_df),
        diverse=True,
        seed=train_cfg.get('seed', 42)
    )
    
    nf, pw, targets, fn, aa = generate_training_data(nodes_df, profiles, train_cfg.get('decay_constant', 2000.0))
    
    normalizer = FeatureNormalizer()
    nf_norm = normalizer.fit_transform(nf, fn)
    t_min, t_max = targets.min(), targets.max()
    t_norm = (targets - t_min) / (t_max - t_min + 1e-8)
    
    model = PNMLRModel(
        n_node_features=nf_norm.shape[1],
        n_amenity_types=pw.shape[1],
        hidden_dims=tuple(train_cfg.get('hidden_dims', [64, 32])),
        learning_rate=train_cfg.get('learning_rate', 0.001),
        seed=train_cfg.get('seed', 42)
    )
    
    model.fit(nf_norm, pw, t_norm, epochs=train_cfg.get('epochs', 100), batch_size=train_cfg.get('batch_size', 256), verbose=True)
    
    model.save(output_dir / "pnmlr_model.pkl")
    normalizer.save(output_dir / "pnmlr_normalizer.json")
    profiles.save(output_dir / "pnmlr_profiles.json")
    
    import json
    (output_dir / "pnmlr_target_params.json").write_text(json.dumps({
        'min': float(t_min), 'max': float(t_max), 'feature_names': fn,
        'amenity_types': aa, 'decay_constant': train_cfg.get('decay_constant', 2000.0)
    }, indent=2))
    
    logger.info(f"Model artifacts saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train PNMLR model")
    parser.add_argument("--city", default="bangalore")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    
    cdm = CityDataManager(args.city, project_root=project_root)
    cfg = cdm.load_config()
    
    train_pnmlr(cdm, cfg)

if __name__ == "__main__":
    main()
