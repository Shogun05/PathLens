#!/usr/bin/env python3
"""
PNMLR Feature Extraction for Amenity Accessibility.

Extracts node-level features for training and inference of the
Personalized Neuro Mixed Logit Regression model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_amenity_distance_columns(df: pd.DataFrame) -> List[str]:
    """Get all amenity distance columns from DataFrame."""
    prefix = "dist_to_"
    return [col for col in df.columns if col.startswith(prefix)]


def get_amenity_names(df: pd.DataFrame) -> List[str]:
    """Extract amenity names from distance columns."""
    prefix = "dist_to_"
    return [col[len(prefix):] for col in get_amenity_distance_columns(df)]


def extract_node_features(
    nodes_df: pd.DataFrame,
    amenity_names: Optional[List[str]] = None,
    include_coords: bool = True,
    include_graph_metrics: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract feature matrix from nodes DataFrame.
    
    Args:
        nodes_df: DataFrame with node attributes
        amenity_names: List of amenity types to include (auto-detect if None)
        include_coords: Whether to include x, y coordinates
        include_graph_metrics: Whether to include degree, travel_time, etc.
    
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if amenity_names is None:
        amenity_names = get_amenity_names(nodes_df)
    
    features = []
    feature_names = []
    
    # Amenity distance features (primary)
    for amenity in amenity_names:
        col = f"dist_to_{amenity}"
        if col in nodes_df.columns:
            values = pd.to_numeric(nodes_df[col], errors='coerce').fillna(np.inf)
            # Replace inf with large value for normalization
            values = values.replace([np.inf, -np.inf], values[np.isfinite(values)].max() * 2)
            features.append(values.values)
            feature_names.append(col)
    
    # Coordinate features
    if include_coords:
        if 'x' in nodes_df.columns:
            features.append(pd.to_numeric(nodes_df['x'], errors='coerce').fillna(0).values)
            feature_names.append('x')
        if 'y' in nodes_df.columns:
            features.append(pd.to_numeric(nodes_df['y'], errors='coerce').fillna(0).values)
            feature_names.append('y')
    
    # Graph metric features
    if include_graph_metrics:
        if 'degree' in nodes_df.columns:
            features.append(pd.to_numeric(nodes_df['degree'], errors='coerce').fillna(0).values)
            feature_names.append('degree')
        if 'travel_time_min' in nodes_df.columns:
            values = pd.to_numeric(nodes_df['travel_time_min'], errors='coerce').fillna(0)
            features.append(values.values)
            feature_names.append('travel_time_min')
        if 'accessibility_score' in nodes_df.columns:
            values = pd.to_numeric(nodes_df['accessibility_score'], errors='coerce').fillna(0)
            features.append(values.values)
            feature_names.append('accessibility_score')
        if 'composite_walkability' in nodes_df.columns:
            values = pd.to_numeric(nodes_df['composite_walkability'], errors='coerce').fillna(0)
            features.append(values.values)
            feature_names.append('composite_walkability')
    
    if not features:
        raise ValueError("No features extracted from nodes DataFrame")
    
    feature_matrix = np.column_stack(features)
    logger.info(f"Extracted {len(feature_names)} features for {len(nodes_df)} nodes")
    
    return feature_matrix, feature_names


class FeatureNormalizer:
    """Min-max feature normalization with persistence."""
    
    def __init__(self):
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FeatureNormalizer':
        """Fit normalizer to feature matrix."""
        self.min_vals = np.nanmin(X, axis=0)
        self.max_vals = np.nanmax(X, axis=0)
        self.feature_names = feature_names
        
        # Handle zero ranges
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0
        self._range_vals = range_vals
        self._fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to [0, 1] range."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        X_norm = (X - self.min_vals) / self._range_vals
        # Clip to handle out-of-range values during inference
        return np.clip(X_norm, 0, 1)
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def save(self, path: Path) -> None:
        """Save normalization parameters to JSON."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        params = {
            'min_vals': self.min_vals.tolist(),
            'max_vals': self.max_vals.tolist(),
            'feature_names': self.feature_names,
        }
        path.write_text(json.dumps(params, indent=2), encoding='utf-8')
        logger.info(f"Saved normalization params to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'FeatureNormalizer':
        """Load normalization parameters from JSON."""
        params = json.loads(path.read_text(encoding='utf-8'))
        normalizer = cls()
        normalizer.min_vals = np.array(params['min_vals'])
        normalizer.max_vals = np.array(params['max_vals'])
        normalizer.feature_names = params.get('feature_names')
        normalizer._range_vals = normalizer.max_vals - normalizer.min_vals
        normalizer._range_vals[normalizer._range_vals == 0] = 1.0
        normalizer._fitted = True
        return normalizer


def compute_profile_utility(
    distances: np.ndarray,
    profile_weights: np.ndarray,
    decay_constant: float = 2000.0,
) -> np.ndarray:
    """
    Compute utility scores for nodes given a user profile.
    
    Uses exponential distance decay: utility = sum(weight * exp(-distance / decay))
    
    Args:
        distances: (N, A) array of distances to A amenity types for N nodes
        profile_weights: (A,) array of user preference weights (sum to 1)
        decay_constant: Distance decay parameter in meters
    
    Returns:
        (N,) array of utility scores
    """
    # Handle infinite distances
    distances = np.where(np.isinf(distances), decay_constant * 10, distances)
    
    # Exponential decay
    decay_scores = np.exp(-distances / decay_constant)
    
    # Weighted sum across amenity types
    utility = np.dot(decay_scores, profile_weights)
    
    return utility
