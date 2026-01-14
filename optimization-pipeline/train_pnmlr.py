#!/usr/bin/env python3
"""
PNMLR Training Script.

Generates training data from synthetic user profiles and trains the PNMLR
model to predict accessibility utility scores.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory for imports
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def generate_training_data(
    nodes_df: pd.DataFrame,
    profiles: UserProfileSet,
    decay_constant: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training data by computing utilities for each node-profile pair.
    
    For each of N nodes and P profiles, compute utility score using
    exponential distance decay weighted by profile preferences.
    
    Args:
        nodes_df: DataFrame with node attributes and distance columns
        profiles: UserProfileSet with synthetic profiles
        decay_constant: Distance decay parameter
    
    Returns:
        Tuple of:
        - node_features: (N*P, F) repeated node features
        - profile_weights: (N*P, A) tiled profile weights
        - targets: (N*P,) utility scores
    """
    # Get amenity distance columns
    amenity_names = profiles.amenity_types
    distance_cols = [f"dist_to_{a}" for a in amenity_names if f"dist_to_{a}" in nodes_df.columns]
    available_amenities = [col.replace("dist_to_", "") for col in distance_cols]
    
    # Filter profiles to only include available amenities
    if len(available_amenities) < len(amenity_names):
        logger.warning(
            f"Some amenities not in data. Using {len(available_amenities)}: {available_amenities}"
        )
        # Reindex profile weights to match available amenities
        amenity_indices = [profiles.amenity_types.index(a) for a in available_amenities]
        profile_weights = profiles.profiles[:, amenity_indices]
        # Renormalize
        profile_weights = profile_weights / profile_weights.sum(axis=1, keepdims=True)
    else:
        profile_weights = profiles.profiles
        available_amenities = amenity_names
    
    # Extract distance matrix
    distances = np.column_stack([
        pd.to_numeric(nodes_df[f"dist_to_{a}"], errors='coerce').fillna(np.inf).values
        for a in available_amenities
    ])
    
    N = len(nodes_df)
    P = profiles.n_profiles
    A = len(available_amenities)
    
    logger.info(f"Generating training data: {N} nodes × {P} profiles = {N*P} samples")
    
    # Compute utility for each node-profile pair
    all_utilities = []
    for p in range(P):
        utilities = compute_profile_utility(distances, profile_weights[p], decay_constant)
        all_utilities.append(utilities)
    
    utilities_matrix = np.column_stack(all_utilities)  # (N, P)
    
    # Flatten for training
    # Each row is a (node, profile) pair
    targets = utilities_matrix.flatten()  # (N*P,)
    
    # Repeat node features P times
    node_features, feature_names = extract_node_features(
        nodes_df, 
        amenity_names=available_amenities,
        include_coords=True,
        include_graph_metrics=True,
    )
    node_features_repeated = np.tile(node_features, (P, 1))  # (N*P, F)
    
    # Tile profile weights N times each
    profile_weights_tiled = np.repeat(profile_weights, N, axis=0)  # (N*P, A)
    
    logger.info(
        f"Training data shape: features={node_features_repeated.shape}, "
        f"profiles={profile_weights_tiled.shape}, targets={targets.shape}"
    )
    
    return node_features_repeated, profile_weights_tiled, targets, feature_names, available_amenities


def train_pnmlr(
    nodes_path: Path,
    output_dir: Path,
    n_profiles: int = 20,
    epochs: int = 100,
    hidden_dims: tuple[int, ...] = (64, 32),
    learning_rate: float = 0.001,
    batch_size: int = 256,
    decay_constant: float = 2000.0,
    seed: int = 42,
) -> tuple[PNMLRModel, UserProfileSet]:
    """
    Train PNMLR model from scratch.
    
    Args:
        nodes_path: Path to nodes_with_scores.parquet
        output_dir: Directory to save model artifacts
        n_profiles: Number of synthetic user profiles
        epochs: Training epochs
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        batch_size: Batch size
        decay_constant: Distance decay for utility computation
        seed: Random seed
    
    Returns:
        Tuple of (trained_model, profiles)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load node data
    logger.info(f"Loading nodes from {nodes_path}")
    nodes_df = pd.read_parquet(nodes_path)
    logger.info(f"Loaded {len(nodes_df)} nodes")
    
    # Generate synthetic user profiles
    logger.info(f"Generating {n_profiles} synthetic user profiles")
    amenity_names = get_amenity_names(nodes_df)
    profiles = UserProfileSet.generate(
        n_profiles=n_profiles,
        amenity_types=amenity_names,
        diverse=True,
        seed=seed,
    )
    logger.info(profiles.describe())
    
    # Generate training data
    node_features, profile_weights, targets, feature_names, available_amenities = generate_training_data(
        nodes_df, profiles, decay_constant
    )
    
    # Normalize features
    normalizer = FeatureNormalizer()
    node_features_norm = normalizer.fit_transform(node_features, feature_names)
    
    # Normalize targets to [0, 1] for stable training
    target_min = targets.min()
    target_max = targets.max()
    targets_norm = (targets - target_min) / (target_max - target_min + 1e-8)
    
    # Create and train model
    n_node_features = node_features_norm.shape[1]
    n_amenity_types = profile_weights.shape[1]
    
    logger.info(
        f"Creating PNMLR model: {n_node_features} node features, "
        f"{n_amenity_types} amenity types, hidden={hidden_dims}"
    )
    
    model = PNMLRModel(
        n_node_features=n_node_features,
        n_amenity_types=n_amenity_types,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        seed=seed,
    )
    
    logger.info(f"Training for {epochs} epochs...")
    model.fit(
        node_features_norm,
        profile_weights,
        targets_norm,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )
    
    # Save artifacts
    model_path = output_dir / "pnmlr_model.pkl"
    model.save(model_path)
    
    normalizer_path = output_dir / "pnmlr_normalizer.json"
    normalizer.save(normalizer_path)
    
    profiles_path = output_dir / "pnmlr_profiles.json"
    profiles.save(profiles_path)
    
    # Save target normalization params
    target_params = {
        'min': float(target_min),
        'max': float(target_max),
        'feature_names': feature_names,
        'amenity_types': available_amenities,
        'decay_constant': decay_constant,
    }
    target_params_path = output_dir / "pnmlr_target_params.json"
    import json
    target_params_path.write_text(json.dumps(target_params, indent=2), encoding='utf-8')
    
    logger.info(f"Saved model artifacts to {output_dir}")
    logger.info(f"  - Model: {model_path}")
    logger.info(f"  - Normalizer: {normalizer_path}")
    logger.info(f"  - Profiles: {profiles_path}")
    logger.info(f"  - Target params: {target_params_path}")
    
    # Validation: predict on original data
    logger.info("Validating model predictions...")
    node_features_single, _ = extract_node_features(nodes_df, amenity_names=available_amenities)
    node_features_single_norm = normalizer.transform(node_features_single)
    
    avg_utility = model.predict_average_utility(
        node_features_single_norm,
        profiles.get_all_profiles()[:, [profiles.amenity_types.index(a) for a in available_amenities]]
    )
    
    logger.info(
        f"Average utility range: [{avg_utility.min():.4f}, {avg_utility.max():.4f}], "
        f"mean={avg_utility.mean():.4f}"
    )
    
    return model, profiles


def main():
    parser = argparse.ArgumentParser(
        description="Train PNMLR model for accessibility utility prediction"
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "analysis" / "nodes_with_scores.parquet",
        help="Path to nodes_with_scores.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Output directory for model artifacts",
    )
    parser.add_argument("--n-profiles", type=int, default=20, help="Number of synthetic profiles")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dims", type=str, default="64,32", help="Hidden layer dims (comma-separated)")
    parser.add_argument("--decay", type=float, default=2000.0, help="Distance decay constant")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    hidden_dims = tuple(int(d) for d in args.hidden_dims.split(","))
    
    train_pnmlr(
        nodes_path=args.nodes,
        output_dir=args.output,
        n_profiles=args.n_profiles,
        epochs=args.epochs,
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        decay_constant=args.decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
