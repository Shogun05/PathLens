#!/usr/bin/env python3
"""
Synthetic User Profile Generation for PNMLR.

Generates heterogeneous accessibility preference profiles using Dirichlet
sampling to capture diverse but plausible preference patterns.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default amenity types matching PathLens config
DEFAULT_AMENITY_TYPES = [
    'school', 'hospital', 'pharmacy', 'supermarket', 'bus_station', 'park', 'bank'
]

# Dirichlet concentration parameters for different user archetypes
# Higher values = more uniform, lower values = more peaked
ARCHETYPE_CONCENTRATIONS = {
    'balanced': 2.0,      # Relatively uniform preferences
    'focused': 0.5,       # Strong preference for 1-2 amenities
    'moderate': 1.0,      # Moderate variation
}


def generate_dirichlet_profiles(
    n_profiles: int = 20,
    amenity_types: Optional[List[str]] = None,
    concentration: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate synthetic user profiles using Dirichlet distribution.
    
    Each profile is a normalized weight vector over amenity types,
    representing heterogeneous accessibility preferences.
    
    Args:
        n_profiles: Number of profiles to generate
        amenity_types: List of amenity type names
        concentration: Dirichlet concentration parameter (alpha)
                      Lower = more peaked, Higher = more uniform
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (profiles_matrix, amenity_types)
        - profiles_matrix: (n_profiles, n_amenities) array of weights
        - amenity_types: List of amenity names
    """
    if amenity_types is None:
        amenity_types = DEFAULT_AMENITY_TYPES
    
    n_amenities = len(amenity_types)
    
    if seed is not None:
        np.random.seed(seed)
    
    # Use Dirichlet distribution for valid probability vectors
    alpha = np.ones(n_amenities) * concentration
    profiles = np.random.dirichlet(alpha, size=n_profiles)
    
    logger.info(
        f"Generated {n_profiles} synthetic user profiles over {n_amenities} amenities "
        f"(concentration={concentration})"
    )
    
    return profiles, amenity_types


def generate_diverse_profiles(
    n_profiles: int = 20,
    amenity_types: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate diverse profiles mixing different concentration levels.
    
    Creates a mix of:
    - Balanced users (roughly equal preferences)
    - Focused users (strong preference for specific amenities)
    - Moderate users (in-between)
    
    Args:
        n_profiles: Total number of profiles
        amenity_types: List of amenity type names
        seed: Random seed
    
    Returns:
        Tuple of (profiles_matrix, amenity_types)
    """
    if amenity_types is None:
        amenity_types = DEFAULT_AMENITY_TYPES
    
    n_amenities = len(amenity_types)
    
    if seed is not None:
        np.random.seed(seed)
    
    profiles = []
    
    # Distribute profiles across archetypes
    n_balanced = n_profiles // 3
    n_focused = n_profiles // 3
    n_moderate = n_profiles - n_balanced - n_focused
    
    # Balanced profiles
    alpha_balanced = np.ones(n_amenities) * ARCHETYPE_CONCENTRATIONS['balanced']
    profiles.extend(np.random.dirichlet(alpha_balanced, size=n_balanced).tolist())
    
    # Focused profiles (strong preference for specific amenities)
    for i in range(n_focused):
        # Create peaked alpha: one amenity gets higher weight
        alpha_focused = np.ones(n_amenities) * ARCHETYPE_CONCENTRATIONS['focused']
        peak_idx = i % n_amenities
        alpha_focused[peak_idx] = 3.0  # Higher concentration for one amenity
        profile = np.random.dirichlet(alpha_focused)
        profiles.append(profile.tolist())
    
    # Moderate profiles
    alpha_moderate = np.ones(n_amenities) * ARCHETYPE_CONCENTRATIONS['moderate']
    profiles.extend(np.random.dirichlet(alpha_moderate, size=n_moderate).tolist())
    
    profiles_array = np.array(profiles)
    
    # Shuffle to mix archetypes
    np.random.shuffle(profiles_array)
    
    logger.info(
        f"Generated {n_profiles} diverse profiles: "
        f"{n_balanced} balanced, {n_focused} focused, {n_moderate} moderate"
    )
    
    return profiles_array, amenity_types


class UserProfileSet:
    """Container for synthetic user profiles with persistence."""
    
    def __init__(
        self,
        profiles: np.ndarray,
        amenity_types: List[str],
        metadata: Optional[Dict] = None,
    ):
        self.profiles = profiles
        self.amenity_types = amenity_types
        self.n_profiles = profiles.shape[0]
        self.n_amenities = profiles.shape[1]
        self.metadata = metadata or {}
    
    def get_profile(self, idx: int) -> np.ndarray:
        """Get a single profile by index."""
        return self.profiles[idx]
    
    def get_all_profiles(self) -> np.ndarray:
        """Get all profiles as (n_profiles, n_amenities) array."""
        return self.profiles
    
    def save(self, path: Path) -> None:
        """Save profiles to JSON."""
        data = {
            'profiles': self.profiles.tolist(),
            'amenity_types': self.amenity_types,
            'n_profiles': self.n_profiles,
            'metadata': self.metadata,
        }
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        logger.info(f"Saved {self.n_profiles} user profiles to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'UserProfileSet':
        """Load profiles from JSON."""
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls(
            profiles=np.array(data['profiles']),
            amenity_types=data['amenity_types'],
            metadata=data.get('metadata', {}),
        )
    
    @classmethod
    def generate(
        cls,
        n_profiles: int = 20,
        amenity_types: Optional[List[str]] = None,
        diverse: bool = True,
        seed: int = 42,
    ) -> 'UserProfileSet':
        """Factory method to generate profiles."""
        if diverse:
            profiles, amenity_types = generate_diverse_profiles(
                n_profiles=n_profiles,
                amenity_types=amenity_types,
                seed=seed,
            )
        else:
            profiles, amenity_types = generate_dirichlet_profiles(
                n_profiles=n_profiles,
                amenity_types=amenity_types,
                seed=seed,
            )
        
        return cls(
            profiles=profiles,
            amenity_types=amenity_types,
            metadata={
                'generation_method': 'diverse' if diverse else 'dirichlet',
                'seed': seed,
            },
        )
    
    def describe(self) -> str:
        """Get human-readable description of profiles."""
        lines = [
            f"UserProfileSet: {self.n_profiles} profiles over {self.n_amenities} amenities",
            f"Amenities: {', '.join(self.amenity_types)}",
            "",
            "Profile weight statistics:",
        ]
        
        for i, amenity in enumerate(self.amenity_types):
            weights = self.profiles[:, i]
            lines.append(
                f"  {amenity:15s}: mean={weights.mean():.3f}, "
                f"std={weights.std():.3f}, range=[{weights.min():.3f}, {weights.max():.3f}]"
            )
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo: generate and display profiles
    logging.basicConfig(level=logging.INFO)
    
    profile_set = UserProfileSet.generate(n_profiles=20, seed=42)
    print(profile_set.describe())
    
    print("\nSample profiles:")
    for i in range(min(3, profile_set.n_profiles)):
        print(f"  Profile {i}: {profile_set.get_profile(i).round(3)}")
