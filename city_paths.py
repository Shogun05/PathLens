#!/usr/bin/env python3
"""
Centralized city data path management for PathLens.

This module provides a unified interface for accessing city-specific data paths,
eliminating hardcoded paths and enabling modular multi-city support.
"""
from pathlib import Path
from typing import Optional
import os
import yaml
from datetime import datetime


# City name normalization and aliases
CITY_ALIASES = {
    "bengaluru": "bangalore",
    "bombay": "mumbai",
    "new_mumbai": "navi_mumbai",
}


def normalize_city_name(city: str) -> str:
    """
    Normalize city name and resolve aliases.
    
    Args:
        city: Raw city name (e.g., "Bengaluru", "NEW MUMBAI", "mumbai")
    
    Returns:
        Normalized city name (e.g., "bangalore", "navi_mumbai", "mumbai")
    """
    normalized = city.lower().strip().replace(" ", "_")
    return CITY_ALIASES.get(normalized, normalized)


class CityDataManager:
    """
    Manages all data paths for a specific city.
    
    Provides a centralized, type-safe interface for accessing city-specific
    data directories and files, enabling modular multi-city support.
    
    Example:
        >>> cdm = CityDataManager("Mumbai")
        >>> print(cdm.baseline_nodes)
        data/cities/mumbai/baseline/nodes_with_scores.parquet
        
        >>> cdm = CityDataManager("bangalore", mode="ga_milp_pnmlr")
        >>> print(cdm.optimized_nodes())
        data/cities/bangalore/optimized/ga_milp_pnmlr/nodes_with_scores.parquet
    """
    
    def __init__(
        self, 
        city_name: str, 
        project_root: Optional[Path] = None,
        mode: str = "ga_only"
    ):
        """
        Initialize city data manager.
        
        Args:
            city_name: Name of the city (will be normalized)
            project_root: Project root directory (auto-detected if None)
            mode: Default optimization mode for path resolution
        """
        self.city = normalize_city_name(city_name)
        
        if project_root is None:
            # Auto-detect project root (assumes this file is in project root)
            project_root = Path(__file__).resolve().parent
        
        self.project_root = Path(project_root)
        self.root = self.project_root / "data" / "cities" / self.city
        self.default_mode = mode
        
        # Ensure city directory exists
        self.root.mkdir(parents=True, exist_ok=True)
        self._config_cache = None

    def load_config(self, force_refresh: bool = False) -> dict:
        """
        Load and merge global and city-specific configurations.
        
        Args:
            force_refresh: Whether to reload from disk if already cached.
            
        Returns:
            Merged configuration dictionary.
        """
        if self._config_cache is not None and not force_refresh:
            return self._config_cache

        # Load base config
        base_path = self.global_config
        base_cfg = {}
        if base_path.exists():
            with open(base_path, 'r') as f:
                base_cfg = yaml.safe_load(f) or {}

        # Load city-specific override if it exists
        city_path = self.city_config
        city_cfg = {}
        if city_path and city_path.exists():
            with open(city_path, 'r') as f:
                city_cfg = yaml.safe_load(f) or {}

        self._config_cache = self.merge_configs(base_cfg, city_cfg)
        return self._config_cache

    @staticmethod
    def merge_configs(base: dict, override: dict) -> dict:
        """Recursively merge two dictionaries."""
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = CityDataManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    # ========================================
    # Raw Data Paths
    # ========================================
    
    @property
    def raw_dir(self) -> Path:
        """OSM raw data directory."""
        path = self.root / "raw" / "osm"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def raw_graph(self) -> Path:
        """Raw OSM road network graph (GraphML format)."""
        return self.raw_dir / "graph.graphml"
    
    @property
    def raw_pois(self) -> Path:
        """Raw POIs GeoJSON file."""
        return self.raw_dir / "pois.geojson"
    
    @property
    def raw_amenities(self) -> Path:
        """Raw amenities GeoJSON file (alternative to pois)."""
        return self.raw_dir / "amenities.geojson"
    
    @property
    def raw_buildings(self) -> Path:
        """Raw buildings GeoJSON file."""
        return self.raw_dir / "buildings.geojson"
    
    @property
    def raw_landuse(self) -> Path:
        """Raw landuse GeoJSON file."""
        return self.raw_dir / "landuse.geojson"
    
    # ========================================
    # Processed Data Paths
    # ========================================
    
    @property
    def processed_dir(self) -> Path:
        """Processed data directory."""
        path = self.root / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def processed_graph(self) -> Path:
        """Processed road network graph (GraphML)."""
        return self.processed_dir / "graph.graphml"
    
    @property
    def poi_mapping(self) -> Path:
        """POI to node mapping (Parquet)."""
        return self.processed_dir / "poi_node_mapping.parquet"
    
    # ========================================
    # Baseline Data Paths
    # ========================================
    
    @property
    def baseline_dir(self) -> Path:
        """Baseline analysis directory."""
        path = self.root / "baseline"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def baseline_nodes(self) -> Path:
        """Baseline nodes with scores (Parquet)."""
        return self.baseline_dir / "nodes_with_scores.parquet"
    
    @property
    def baseline_nodes_csv(self) -> Path:
        """Baseline nodes with scores (CSV)."""
        return self.baseline_dir / "nodes_with_scores.csv"
    
    @property
    def baseline_h3(self) -> Path:
        """Baseline H3 aggregations (Parquet)."""
        return self.baseline_dir / "h3_agg.parquet"
    
    @property
    def baseline_h3_csv(self) -> Path:
        """Baseline H3 aggregations (CSV)."""
        return self.baseline_dir / "h3_agg.csv"
    
    @property
    def baseline_metrics(self) -> Path:
        """Baseline metrics summary (JSON)."""
        return self.baseline_dir / "metrics_summary.json"
    
    @property
    def baseline_pois_parquet(self) -> Path:
        """Baseline POIs (Parquet format, for fast loading)."""
        return self.baseline_dir / "pois.parquet"
    
    # ========================================
    # Optimized Data Paths
    # ========================================
    
    def optimized_dir(self, mode: Optional[str] = None) -> Path:
        """
        Optimized data directory for a specific mode.
        
        Args:
            mode: Optimization mode (ga_only, ga_milp, ga_milp_pnmlr)
                  Defaults to the mode specified in __init__
        """
        mode = mode or self.default_mode
        path = self.root / "optimized" / mode
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def optimized_nodes(self, mode: Optional[str] = None) -> Path:
        """Optimized nodes with scores (Parquet)."""
        return self.optimized_dir(mode) / "nodes_with_scores.parquet"
    
    def optimized_nodes_csv(self, mode: Optional[str] = None) -> Path:
        """Optimized nodes with scores (CSV)."""
        return self.optimized_dir(mode) / "nodes_with_scores.csv"
    
    def optimized_h3(self, mode: Optional[str] = None) -> Path:
        """Optimized H3 aggregations (Parquet)."""
        return self.optimized_dir(mode) / "h3_agg.parquet"
    
    def optimized_h3_csv(self, mode: Optional[str] = None) -> Path:
        """Optimized H3 aggregations (CSV)."""
        return self.optimized_dir(mode) / "h3_agg.csv"
    
    def optimized_metrics(self, mode: Optional[str] = None) -> Path:
        """Optimized metrics summary (JSON). Uses mode-prefixed filename if mode is specified."""
        m = mode or self.mode or "ga_only"
        return self.optimized_dir(mode) / f"{m}_metrics_summary.json"
    
    def optimized_pois(self, mode: Optional[str] = None) -> Path:
        """Optimized POIs GeoJSON."""
        return self.optimized_dir(mode) / "optimized_pois.geojson"
    
    def best_candidate(self, mode: Optional[str] = None) -> Path:
        """Best candidate solution (JSON)."""
        return self.optimized_dir(mode) / "best_candidate.json"
    
    def high_travel_nodes(self, mode: Optional[str] = None) -> Path:
        """High travel time nodes for optimization (CSV)."""
        return self.optimized_dir(mode) / "high_travel_time_nodes.csv"
    
    def optimization_map(self, mode: Optional[str] = None) -> Path:
        """Optimization visualization HTML map."""
        return self.optimized_dir(mode) / "optimized_map.html"
    
    def combined_pois(self, mode: Optional[str] = None) -> Path:
        """Combined baseline + optimized POIs (GeoJSON)."""
        return self.optimized_dir(mode) / "combined_pois.geojson"
    
    # ========================================
    # Configuration
    # ========================================
    
    @property
    def city_config(self) -> Optional[Path]:
        """
        City-specific config file (if exists).
        Checks in configs/{city}.yaml first, then falls back to city root/config.yaml.
        """
        # Check orchestration config first
        orchestration_config = self.project_root / "configs" / f"{self.city}.yaml"
        if orchestration_config.exists():
            return orchestration_config
            
        # Fallback to local snapshot
        config_path = self.root / "config.yaml"
        return config_path if config_path.exists() else None
    
    @property
    def global_config(self) -> Path:
        """Global project config file."""
        return self.project_root / "configs" / "base.yaml"
    
    @property
    def config(self) -> Path:
        """
        Active config file (city-specific or global).
        
        Returns city config if it exists, otherwise returns global config.
        """
        return self.city_config or self.global_config
    
    # ========================================
    # Metadata
    # ========================================
    
    @property
    def metadata(self) -> Path:
        """City metadata JSON file."""
        return self.root / "metadata.json"
    
    @property
    def run_summary(self) -> Path:
        """Run summary JSON file."""
        return self.root / "run_summary.json"
    
    # ========================================
    # Satellite Graph (GraphBuilder)
    # ========================================
    
    @property
    def satellite_graph_dir(self) -> Path:
        """Satellite-derived graph directory."""
        path = self.root / "satellite_graph"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # ========================================
    # Landuse Pipeline
    # ========================================
    
    @property
    def landuse_dir(self) -> Path:
        """Landuse pipeline data directory."""
        path = self.root / "landuse"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def feasibility_csv(self, amenity: str) -> Path:
        """Feasibility data for a specific amenity type."""
        return self.landuse_dir / f"feasibility_{amenity}.csv"
    
    def placements_geojson(self, amenity: str) -> Path:
        """GEE placements for a specific amenity type."""
        return self.landuse_dir / f"gee_placements_{amenity}.geojson"
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def exists(self) -> bool:
        """Check if city data directory exists."""
        return self.root.exists()
    
    def has_baseline(self) -> bool:
        """Check if city has baseline data."""
        return self.baseline_nodes.exists() and self.baseline_metrics.exists()
    
    def has_optimized(self, mode: Optional[str] = None) -> bool:
        """Check if city has optimized data for a specific mode."""
        return (
            self.optimized_nodes(mode).exists() 
            and self.optimized_metrics(mode).exists()
        )
    
    def __repr__(self) -> str:
        return f"CityDataManager(city='{self.city}', root='{self.root}')"


# ========================================
# Backward Compatibility Helpers
# ========================================

def get_legacy_analysis_dir(project_root: Optional[Path] = None) -> Path:
    """
    Get legacy global analysis directory.
    
    DEPRECATED: Use CityDataManager instead.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    return Path(project_root) / "data" / "analysis"


def migrate_legacy_to_city(city_name: str, project_root: Optional[Path] = None) -> None:
    """
    Migrate data from legacy global directories to city-specific structure.
    
    This is a helper function to move existing data from:
    - data/analysis/ → data/cities/{city}/baseline/
    - data/raw/osm/ → data/cities/{city}/raw/osm/
    - data/processed/ → data/cities/{city}/processed/
    
    Args:
        city_name: Name of the city to migrate data for
        project_root: Project root directory
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    
    project_root = Path(project_root)
    cdm = CityDataManager(city_name, project_root)
    
    import shutil
    
    # Migrate analysis → baseline
    legacy_analysis = project_root / "data" / "analysis"
    if legacy_analysis.exists():
        print(f"Migrating {legacy_analysis} → {cdm.baseline_dir}")
        for item in legacy_analysis.glob("baseline_*"):
            dest_name = item.name.replace("baseline_", "")
            dest = cdm.baseline_dir / dest_name
            if not dest.exists():
                shutil.copy2(item, dest)
                print(f"  Copied {item.name} → {dest.name}")
    
    # Migrate raw/osm
    legacy_raw = project_root / "data" / "raw" / "osm"
    if legacy_raw.exists() and not cdm.raw_graph.exists():
        print(f"Migrating {legacy_raw} → {cdm.raw_dir}")
        for item in legacy_raw.iterdir():
            dest = cdm.raw_dir / item.name
            if not dest.exists():
                if item.is_file():
                    shutil.copy2(item, dest)
                else:
                    shutil.copytree(item, dest)
                print(f"  Copied {item.name}")
    
    # Migrate processed
    legacy_processed = project_root / "data" / "processed"
    if legacy_processed.exists() and not cdm.processed_graph.exists():
        print(f"Migrating {legacy_processed} → {cdm.processed_dir}")
        for item in legacy_processed.iterdir():
            dest = cdm.processed_dir / item.name
            if not dest.exists():
                if item.is_file():
                    shutil.copy2(item, dest)
                else:
                    shutil.copytree(item, dest)
                print(f"  Copied {item.name}")
    
    print(f"\nMigration complete for {city_name}")
    print(f"Baseline data: {cdm.has_baseline()}")
    print(f"City root: {cdm.root}")


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        if len(sys.argv) < 3:
            print("Usage: python city_paths.py migrate <city_name>")
            sys.exit(1)
        migrate_legacy_to_city(sys.argv[2])
    else:
        # Demo usage
        print("CityDataManager Demo\n" + "=" * 50)
        
        for city in ["bangalore", "mumbai", "chandigarh"]:
            cdm = CityDataManager(city)
            print(f"\n{city.upper()}")
            print(f"  Root: {cdm.root}")
            print(f"  Config: {cdm.config}")
            print(f"  Exists: {cdm.exists()}")
            print(f"  Has baseline: {cdm.has_baseline()}")
            print(f"  Baseline nodes: {cdm.baseline_nodes}")
            print(f"  Optimized (ga_milp): {cdm.optimized_nodes('ga_milp')}")
