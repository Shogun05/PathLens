#!/usr/bin/env python3
"""Hybrid GA runner combining greedy templates with memetic local search.

This script analyses high travel time nodes, inspects amenity distances and runs a
hybrid genetic algorithm that supports pluggable evaluation hooks.

Progress indicators:
 - Uses tqdm if installed to show progress bars for generations and evaluation.
 - Falls back to logging-based progress if tqdm is not available.
 - Writes a heartbeat JSON each generation to analysis_dir/heartbeat.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import yaml
import networkx as nx

# Add project root to path for CityDataManager
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

# Import landuse feasibility integration (optional, lazy-loaded)
try:
    landuse_pipeline_path = Path(__file__).resolve().parent.parent / "landuse-pipeline"
    if str(landuse_pipeline_path) not in sys.path:
        sys.path.insert(0, str(landuse_pipeline_path))
    from run_feasibility import LanduseFeasibilityIntegration
    _HAS_LANDUSE = True
except ImportError:
    LanduseFeasibilityIntegration = None  # type: ignore
    _HAS_LANDUSE = False

# Try to import tqdm for nicer progress bars; gracefully fallback if not present.
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False

# ---------------------------------------------------------------------------
# Data holders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """Encapsulates amenity placements per amenity type."""

    placements: Mapping[str, Tuple[str, ...]]
    template_id: Optional[str] = None

    @property
    def signature(self) -> str:
        ordered = sorted((amenity, tuple(sorted(nodes))) for amenity, nodes in self.placements.items())
        return "|".join(f"{amenity}:{','.join(nodes)}" for amenity, nodes in ordered) or "baseline"


def serialize_candidate(candidate: Candidate) -> Dict[str, object]:
    return {
        "placements": {amenity: list(nodes) for amenity, nodes in candidate.placements.items()},
        "template_id": candidate.template_id,
    }


def deserialize_candidate(payload: Mapping[str, object]) -> Candidate:
    placements: Dict[str, Tuple[str, ...]] = {}
    raw_placements = payload.get("placements", {})
    if isinstance(raw_placements, Mapping):
        for amenity, nodes in raw_placements.items():
            if isinstance(nodes, Iterable):
                placements[str(amenity)] = tuple(sorted(str(node) for node in nodes))
    template_raw = payload.get("template_id")
    template_id = str(template_raw) if template_raw is not None else None
    return Candidate(placements=placements, template_id=template_id)


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _lists_to_tuples(item) for key, item in value.items()}
    return value


@dataclass
class HybridGAConfig:
    """Configuration for Hybrid GA - all values from config YAML."""
    population: int = 80
    generations: int = 50
    crossover_rate: float = 0.75
    mutation_rate: float = 0.2
    elitism: int = 4
    seed_fraction: float = 0.4
    templates: int = 5
    local_search_budget: int = 20
    local_search_topk: int = 5
    per_pool_limit: int = 500
    random_seed: int = 42
    analysis_dir: Path = field(default_factory=lambda: Path("../data/optimization/runs"))
    enable_progress: bool = True
    workers: int = 1


@dataclass
class GAContext:
    nodes: pd.DataFrame
    high_travel_nodes: pd.DataFrame
    amenity_weights: Dict[str, float]
    amenity_pools: Dict[str, List[str]]
    distance_columns: Dict[str, str]
    config: HybridGAConfig
    amenity_budgets: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)
    cdm: Optional[Any] = None  # CityDataManager instance for PNMLR hooks
    raw_config: Dict[str, Any] = field(default_factory=dict)  # Full YAML config dict for hooks


EvaluateHook = Callable[[Candidate, Mapping[str, object], GAContext], Dict[str, object]]
PrecomputeHook = Callable[[GAContext], Mapping[str, object]]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def setup_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def detect_amenity_weights(nodes: pd.DataFrame, cfg: Mapping[str, float], fallback: float = 1.0) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    prefix = "dist_to_"
    for column in nodes.columns:
        if not column.startswith(prefix):
            continue
        amenity = column[len(prefix):].strip()
        if amenity:
            weights[amenity] = float(cfg.get(amenity, fallback))
    if not weights:
        logging.warning("No amenity distance columns detected; defaulting to a single amenity placeholder.")
        weights["amenity"] = float(cfg.get("amenity", fallback))
    return weights


def detect_distance_columns(nodes: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    prefix = "dist_to_"
    for column in nodes.columns:
        if column.startswith(prefix):
            mapping[column[len(prefix):]] = column
    return mapping


def load_callable(spec: Optional[str]) -> Optional[Callable]:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("Hook specification must be in 'module:function' format.")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def ensure_index_on_osmid(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by stringified osmids for consistent lookups."""
    if frame.index.name == "osmid":
        indexed = frame.copy()
    elif "osmid" in frame.columns:
        indexed = frame.set_index("osmid", drop=True)
    else:
        indexed = frame.copy()
    if indexed.index.name != "osmid":
        indexed.index.name = "osmid"
    indexed.index = indexed.index.map(str)
    return indexed


def compute_distance_baselines(nodes: pd.DataFrame, distance_columns: Mapping[str, str]) -> Dict[str, float]:
    baselines: Dict[str, float] = {}
    for amenity, column in distance_columns.items():
        if column not in nodes.columns:
            continue
        series = pd.to_numeric(nodes[column], errors="coerce")
        series = series.replace([math.inf, -math.inf], pd.NA).dropna()
        if series.empty:
            continue
        median_value = float(series.median())
        if math.isnan(median_value) or median_value <= 0:
            mean_value = float(series.mean()) if not series.empty else 0.0
            median_value = mean_value if mean_value > 0 else float(series.max()) if not series.empty else 0.0
        if median_value <= 0:
            median_value = 1.0
        baselines[amenity] = median_value
    return baselines


def analyse_distances(high_nodes: pd.DataFrame, context: GAContext) -> None:
    logging.info("Analysing amenity distances for %d high-travel nodes.", len(high_nodes))
    indexed_nodes = ensure_index_on_osmid(context.nodes)
    common = indexed_nodes.loc[indexed_nodes.index.intersection(high_nodes.index)]
    for amenity, column in context.distance_columns.items():
        if column not in common.columns:
            continue
        series = pd.to_numeric(common[column], errors="coerce")
        mean_dist = float(series.dropna().mean()) if not series.dropna().empty else float("inf")
        logging.info("Amenity %-20s | mean distance %.2f m | available %4d", amenity, mean_dist, series.dropna().shape[0])


def build_amenity_pools(
    high_nodes: pd.DataFrame,
    nodes: pd.DataFrame,
    distance_columns: Mapping[str, str],
    weights: Mapping[str, float],
    per_pool_limit: int,
    min_spacing: int = 1200,
) -> Dict[str, List[str]]:
    """Build candidate pools for each amenity type.
    
    Selects candidates that are FARTHEST from existing amenities of each type,
    making them ideal locations for new amenity placements.
    """
    pools: Dict[str, List[str]] = {}
    base_candidates = ensure_index_on_osmid(high_nodes)
    indexed_nodes = ensure_index_on_osmid(nodes)
    
    center_x = indexed_nodes['x'].median() if 'x' in indexed_nodes.columns else 0
    center_y = indexed_nodes['y'].median() if 'y' in indexed_nodes.columns else 0
    
    for amenity in weights.keys():
        column = distance_columns.get(amenity)
        if column and column in indexed_nodes.columns:
            # Get high-travel-time nodes with their full data
            candidate_frame = indexed_nodes.loc[indexed_nodes.index.intersection(base_candidates.index)].copy()
            
            if column in candidate_frame.columns and not candidate_frame.empty:
                # Get distance values, exclude inf and NaN
                dist_vals = pd.to_numeric(candidate_frame[column], errors='coerce')
                finite_mask = np.isfinite(dist_vals) & dist_vals.notna()
                candidate_frame = candidate_frame[finite_mask].copy()
                
                if not candidate_frame.empty:
                    # Calculate placement score: prioritize far from existing + close to center
                    dist_vals = pd.to_numeric(candidate_frame[column], errors='coerce')
                    
                    # Normalize distance (higher = better for placement)
                    d_min, d_max = dist_vals.min(), dist_vals.max()
                    if d_max > d_min:
                        dist_norm = (dist_vals - d_min) / (d_max - d_min)
                    else:
                        dist_norm = pd.Series(1.0, index=candidate_frame.index)
                    
                    # Calculate distance from city center
                    if 'x' in candidate_frame.columns and 'y' in candidate_frame.columns:
                        center_dist = np.sqrt(
                            (candidate_frame['x'] - center_x)**2 + 
                            (candidate_frame['y'] - center_y)**2
                        )
                        c_min, c_max = center_dist.min(), center_dist.max()
                        if c_max > c_min:
                            center_norm = (center_dist - c_min) / (c_max - c_min)
                        else:
                            center_norm = pd.Series(0.0, index=candidate_frame.index)
                    else:
                        center_norm = pd.Series(0.0, index=candidate_frame.index)
                    
                    # Score: high distance from amenity (0.7) + close to center (0.3)
                    candidate_frame['placement_score'] = 0.7 * dist_norm - 0.3 * center_norm
                    candidates = candidate_frame.sort_values('placement_score', ascending=False)
                else:
                    candidates = candidate_frame
            else:
                candidates = candidate_frame
        else:
            candidates = base_candidates.sort_values("travel_time_min", ascending=False) if "travel_time_min" in base_candidates.columns else base_candidates
        
        node_ids = [str(idx) for idx in candidates.index.tolist()]
        
        # If no valid candidates from high_travel_nodes, fall back to full nodes
        if not node_ids and column and column in indexed_nodes.columns:
            logging.warning("No valid candidates for %s in high-travel nodes, using fallback", amenity)
            dist_vals = pd.to_numeric(indexed_nodes[column], errors='coerce')
            finite_mask = np.isfinite(dist_vals) & dist_vals.notna()
            fallback = indexed_nodes[finite_mask].copy()
            fallback['_sort_dist'] = pd.to_numeric(fallback[column], errors='coerce')
            fallback = fallback.sort_values('_sort_dist', ascending=False)
            node_ids = [str(idx) for idx in fallback.index.tolist()]
        
        if per_pool_limit > 0:
            node_ids = node_ids[:per_pool_limit]
        
        pools[amenity] = node_ids
        logging.info("Amenity %s pool: %d candidates (selected by distance).", amenity, len(node_ids))
    return pools


# ---------------------------------------------------------------------------
# Default evaluation hooks
# ---------------------------------------------------------------------------


def default_precompute_effects(context: GAContext) -> Mapping[str, object]:
    logging.info("Running default precompute for %d nodes.", len(context.nodes))
    indexed_nodes = ensure_index_on_osmid(context.nodes)
    baselines = compute_distance_baselines(indexed_nodes, context.distance_columns)
    context.extra["indexed_nodes"] = indexed_nodes
    context.extra["distance_baselines"] = baselines
    
    graph = context.extra.get("graph")
    graph_undir = None
    if graph is not None:
        logging.info("Precomputing undirected graph for diversity penalty...")
        graph_undir = graph.to_undirected()
    
    return {"nodes": indexed_nodes, "graph": graph, "graph_undir": graph_undir}


def default_evaluate_candidate(candidate: Candidate, effects: Mapping[str, object], context: GAContext) -> Dict[str, object]:
    nodes: pd.DataFrame = effects["nodes"]
    amenity_weights = context.amenity_weights
    total_weight = sum(amenity_weights.values())
    
    total_gain = 0.0
    placements: Dict[str, int] = {}
    best_distances: Dict[str, float] = {}
    amenity_scores: Dict[str, float] = {}

    for amenity, node_ids in candidate.placements.items():
        column = context.distance_columns.get(amenity)
        if not column or column not in nodes.columns:
            continue
        target_nodes = [str(node) for node in node_ids]
        distances = nodes.loc[nodes.index.isin(target_nodes), column]
        weight = amenity_weights.get(amenity, 1.0)
        if distances.empty:
            continue
        numeric_distances = pd.to_numeric(distances, errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
        if numeric_distances.empty:
            continue
        min_distance = float(numeric_distances.min())
        best_distances[amenity] = min_distance
        
        gain_per_node = 0.0
        for dist in numeric_distances:
            if dist >= 1200:
                gain_per_node += 1.0
            else:
                gain_per_node += (dist / 1200.0) * 0.5
        
        score = gain_per_node * (weight / total_weight) * 10.0
        amenity_scores[amenity] = score
        total_gain += score
        placements[amenity] = len(target_nodes)

    travel_penalty = 0.0
    if "travel_time_min" in nodes.columns:
        placed_index = [str(node) for ids in candidate.placements.values() for node in ids]
        placed_nodes = nodes.loc[nodes.index.isin(placed_index)]
        if not placed_nodes.empty:
            travel_penalty = float(placed_nodes["travel_time_min"].mean())

    diversity_penalty = 0.0
    proximity_penalty = 0.0
    
    for amenity, node_ids in candidate.placements.items():
        column = context.distance_columns.get(amenity)
        if not column or column not in nodes.columns:
            continue
        for node in node_ids:
            node_str = str(node)
            if node_str in nodes.index:
                existing_dist = pd.to_numeric(nodes.loc[node_str, column], errors='coerce')
                if pd.notna(existing_dist) and existing_dist < 600:
                    ratio = existing_dist / 600.0
                    penalty = (1.0 - ratio) ** 2
                    proximity_penalty += penalty * 0.5
    
    for amenity, node_ids in candidate.placements.items():
        if len(node_ids) <= 1:
            continue
        coords = []
        for node in node_ids:
            node_str = str(node)
            if node_str in nodes.index and 'x' in nodes.columns and 'y' in nodes.columns:
                x = nodes.loc[node_str, 'x']
                y = nodes.loc[node_str, 'y']
                if pd.notna(x) and pd.notna(y):
                    coords.append((float(x), float(y)))
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist_meters = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                min_spacing = 800.0
                if dist_meters < min_spacing:
                    ratio = dist_meters / min_spacing
                    penalty = (1.0 - ratio) ** 2
                    diversity_penalty += penalty * 2.0

    fitness = total_gain - 0.0005 * travel_penalty - diversity_penalty - proximity_penalty
    fitness = max(fitness, 0.0)
    return {
        "fitness": fitness,
        "distance_gain": total_gain,
        "travel_penalty": travel_penalty,
        "diversity_penalty": diversity_penalty,
        "proximity_penalty": proximity_penalty,
        "placements": placements,
        "best_distances": best_distances,
        "amenity_scores": amenity_scores,
    }


# ---------------------------------------------------------------------------
# GA implementation
# ---------------------------------------------------------------------------


class HybridGA:
    def __init__(
        self,
        context: GAContext,
        config: HybridGAConfig,
        precompute_hook: PrecomputeHook,
        evaluate_hook: EvaluateHook,
        enable_landuse_filter: bool = False,
        cdm: Optional[Any] = None,
        raw_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.context = context
        self.config = config
        self.precompute_hook = precompute_hook
        self.evaluate_hook = evaluate_hook
        self.effects: Mapping[str, object] = {}
        self.random = random.Random(config.random_seed)
        self.operator_credits: MutableMapping[str, int] = {"crossover": 0, "mutation": 0, "local_search": 0}
        self.analysis_dir = config.analysis_dir
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.landuse_filter = None
        if enable_landuse_filter:
            if _HAS_LANDUSE and LanduseFeasibilityIntegration is not None and cdm is not None and raw_config is not None:
                logging.info("Initializing landuse feasibility filter...")
                self.landuse_filter = LanduseFeasibilityIntegration(cdm, raw_config)
                logging.info("Landuse feasibility filter initialized.")
            else:
                logging.warning("Landuse filtering requested but not available (missing cdm or config).")

    def seed_population(self) -> List[Candidate]:
        population: List[Candidate] = []
        template_candidates = self.build_greedy_templates(self.config.templates)
        num_seeded = max(1, int(self.config.population * self.config.seed_fraction))
        population.extend(template_candidates[:num_seeded])
        logging.info("Seeded population with %d greedy templates.", len(population))
        while len(population) < self.config.population:
            population.append(self.random_candidate())
        self.random.shuffle(population)
        return population

    def build_greedy_templates(self, max_templates: int) -> List[Candidate]:
        templates: List[Candidate] = []
        sorted_amenities = sorted(self.context.amenity_weights.items(), key=lambda item: item[1], reverse=True)
        pools = self.context.amenity_pools
        for template_idx in range(max_templates):
            placements: Dict[str, Tuple[str, ...]] = {}
            for amenity, _ in sorted_amenities:
                pool = pools.get(amenity, [])
                if not pool:
                    continue
                max_budget = self.context.amenity_budgets.get(amenity, 20)
                min_placements = max(3, max_budget // 2)
                num_to_place = min(max_budget, min_placements + template_idx % (max_budget - min_placements + 1))
                num_to_place = min(num_to_place, len(pool))
                slice_start = min(template_idx * 2, max(0, len(pool) - num_to_place))
                slice_end = min(slice_start + num_to_place, len(pool))
                placements[amenity] = tuple(pool[slice_start:slice_end])
            templates.append(Candidate(placements=placements, template_id=f"greedy_{template_idx}"))
        return templates

    def random_candidate(self) -> Candidate:
        placements: Dict[str, Tuple[str, ...]] = {}
        for amenity, pool in self.context.amenity_pools.items():
            if not pool:
                continue
            max_budget = self.context.amenity_budgets.get(amenity, 20)
            sample_pct = 0.10 + (self.random.random() * 0.05)
            sample_size = min(max_budget, max(5, int(len(pool) * sample_pct)))
            sample_size = min(sample_size, len(pool))
            chosen = self.random.sample(pool, k=sample_size)
            placements[amenity] = tuple(sorted(chosen))
        return Candidate(placements=placements)

    @property
    def checkpoint_path(self) -> Path:
        return self.analysis_dir / "checkpoint.json"

    def _load_checkpoint(self) -> Optional[Dict[str, object]]:
        path = self.checkpoint_path
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data: Dict[str, object] = json.load(handle)
            logging.info("Loaded checkpoint from %s", path)
            return data
        except Exception as exc:
            logging.warning("Failed to load checkpoint %s: %s", path, exc)
            return None

    def _save_checkpoint(self, next_generation: int, population: Sequence[Candidate], history: Sequence[Dict[str, object]], best_overall: Optional[Tuple[Candidate, Dict[str, object]]]) -> None:
        payload: Dict[str, object] = {
            "version": 1,
            "timestamp": time.time(),
            "next_generation": int(next_generation),
            "population": [serialize_candidate(c) for c in population],
            "history": list(history),
            "operator_credits": dict(self.operator_credits),
            "random_state": self.random.getstate(),
        }
        if best_overall:
            payload["best_overall"] = {"candidate": serialize_candidate(best_overall[0]), "metrics": best_overall[1]}
        try:
            self.checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logging.warning("Failed to persist checkpoint: %s", exc)

    def _clear_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
            except Exception:
                pass

    def crossover(self, a: Candidate, b: Candidate) -> Tuple[Candidate, Candidate]:
        if self.random.random() >= self.config.crossover_rate:
            return a, b
        child_a: Dict[str, Tuple[str, ...]] = {}
        child_b: Dict[str, Tuple[str, ...]] = {}
        for amenity in set(a.placements.keys()) | set(b.placements.keys()):
            nodes_a = set(a.placements.get(amenity, ()))
            nodes_b = set(b.placements.get(amenity, ()))
            union = list(nodes_a | nodes_b)
            if not union:
                continue
            self.random.shuffle(union)
            pivot = self.random.randint(0, len(union))
            child_a[amenity] = tuple(sorted(union[:pivot]))
            child_b[amenity] = tuple(sorted(union[pivot:]))
        self.operator_credits["crossover"] += 1
        return Candidate(child_a), Candidate(child_b)

    def mutate(self, candidate: Candidate) -> Candidate:
        placements: Dict[str, Tuple[str, ...]] = {}
        for amenity, nodes in candidate.placements.items():
            if self.random.random() >= self.config.mutation_rate:
                placements[amenity] = nodes
                continue
            pool = self.context.amenity_pools.get(amenity, [])
            if not pool:
                placements[amenity] = nodes
                continue
            current = set(nodes)
            if current and self.random.random() < 0.5:
                try:
                    current.remove(self.random.choice(tuple(current)))
                except KeyError:
                    pass
            else:
                available = list(set(pool) - current)
                if available:
                    current.add(self.random.choice(available))
            placements[amenity] = tuple(sorted(current))
            self.operator_credits["mutation"] += 1
        return Candidate(placements=placements)

    def evaluate(self, candidate: Candidate) -> Dict[str, object]:
        return self.evaluate_hook(candidate, self.effects, self.context)

    def _evaluate_population(self, population: Sequence[Candidate], generation: int, sink: List[Tuple[Candidate, Dict[str, object]]]) -> None:
        if self.config.workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                futures = {executor.submit(self.evaluate, c): c for c in population}
                for future in as_completed(futures):
                    sink.append((futures[future], future.result()))
        else:
            iterator = tqdm(population, desc=f"Gen {generation} eval", leave=False) if _HAS_TQDM and self.config.enable_progress else population
            for candidate in iterator:
                sink.append((candidate, self.evaluate(candidate)))

    def local_search(self, candidate: Candidate, base_metrics: Dict[str, object]) -> Tuple[Candidate, Dict[str, object]]:
        best_candidate = candidate
        best_metrics = base_metrics
        best_fitness = float(base_metrics.get("fitness", float("-inf")))
        for _ in range(self.config.local_search_budget):
            improved = False
            for amenity, pool in self.context.amenity_pools.items():
                current_nodes = set(best_candidate.placements.get(amenity, ()))
                for node in pool[:50]:
                    if node in current_nodes:
                        continue
                    trial_nodes = tuple(sorted(current_nodes | {node}))
                    trial_placements = dict(best_candidate.placements)
                    trial_placements[amenity] = trial_nodes
                    trial_candidate = Candidate(placements=trial_placements, template_id=best_candidate.template_id)
                    trial_metrics = self.evaluate(trial_candidate)
                    trial_fitness = float(trial_metrics.get("fitness", float("-inf")))
                    if trial_fitness > best_fitness:
                        best_candidate, best_metrics, best_fitness = trial_candidate, trial_metrics, trial_fitness
                        improved = True
                        self.operator_credits["local_search"] += 1
                        break
                if improved:
                    break
            if not improved:
                break
        return best_candidate, best_metrics

    def _write_heartbeat(self, generation: int, best_metrics: Dict[str, object], elapsed: float) -> None:
        try:
            heartbeat = {"generation": generation, "best_fitness": float(best_metrics.get("fitness", 0.0)), "elapsed_seconds": elapsed, "timestamp": time.time()}
            (self.analysis_dir / "heartbeat.json").write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")
        except Exception:
            pass

    def run(self) -> Dict[str, object]:
        logging.info("Starting hybrid GA: population=%d, generations=%d", self.config.population, self.config.generations)
        start_run = time.time()
        self.effects = self.precompute_hook(self.context)
        
        resume_data = self._load_checkpoint()
        history: List[Dict[str, object]] = []
        best_overall: Optional[Tuple[Candidate, Dict[str, object]]] = None
        start_generation = 1
        
        if resume_data:
            start_generation = max(1, int(resume_data.get("next_generation", 1)))
            population_payload = resume_data.get("population", [])
            population = [deserialize_candidate(p) for p in population_payload if isinstance(p, Mapping)]
            if not population:
                population = self.seed_population()
                start_generation = 1
        else:
            population = self.seed_population()

        gen_range = range(start_generation, self.config.generations + 1)
        gen_iterator = tqdm(gen_range, desc="Generations", unit="gen") if _HAS_TQDM and self.config.enable_progress else gen_range

        for generation in gen_iterator:
            gen_start = time.time()
            evaluated: List[Tuple[Candidate, Dict[str, object]]] = []
            self._evaluate_population(population, generation, evaluated)
            
            if not evaluated:
                continue
            
            evaluated.sort(key=lambda item: float(item[1].get("fitness", float("-inf"))), reverse=True)
            
            # Local search on top-k
            enhanced = []
            for candidate, metrics in evaluated[:self.config.local_search_topk]:
                enhanced.append(self.local_search(candidate, metrics))
            evaluated[:self.config.local_search_topk] = enhanced
            evaluated.sort(key=lambda item: float(item[1].get("fitness", float("-inf"))), reverse=True)

            best_candidate, best_metrics = evaluated[0]
            gen_elapsed = time.time() - gen_start
            logging.info("Gen %d best fitness %.4f | placements=%s | elapsed=%.2fs", generation, float(best_metrics.get("fitness", 0.0)), best_metrics.get("placements"), gen_elapsed)

            if best_overall is None or float(best_metrics.get("fitness", float("-inf"))) > float(best_overall[1].get("fitness", float("-inf"))):
                best_overall = (best_candidate, best_metrics)
                self.persist_best(generation, best_candidate, best_metrics)
            self._write_heartbeat(generation, best_metrics, gen_elapsed)

            history.append({"generation": generation, "best_fitness": float(best_metrics.get("fitness", 0.0))})

            # Next generation
            elites = [item[0] for item in evaluated[:self.config.elitism]]
            next_population: List[Candidate] = elites.copy()
            while len(next_population) < self.config.population:
                parent_a = self.tournament_select(evaluated)
                parent_b = self.tournament_select(evaluated)
                child_a, child_b = self.crossover(parent_a, parent_b)
                next_population.append(self.mutate(child_a))
                if len(next_population) < self.config.population:
                    next_population.append(self.mutate(child_b))
            
            self._save_checkpoint(generation + 1, next_population, history, best_overall)
            population = next_population

        assert best_overall is not None
        result = {"best_candidate": best_overall[0].signature, "best_metrics": best_overall[1], "history": history}
        (self.analysis_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logging.info("GA complete. Total elapsed %.2fs", time.time() - start_run)
        self._clear_checkpoint()
        return result

    def tournament_select(self, evaluated: Sequence[Tuple[Candidate, Dict[str, object]]], k: int = 3) -> Candidate:
        participants = self.random.sample(evaluated, k=min(k, len(evaluated)))
        participants.sort(key=lambda item: float(item[1].get("fitness", float("-inf"))), reverse=True)
        return participants[0][0]

    def filter_conflicts(self, candidate: Candidate) -> Candidate:
        """Remove nodes violating spacing constraints."""
        placements: Dict[str, Tuple[str, ...]] = {}
        nodes = self.context.nodes
        min_spacing = 1200.0
        
        if 'x' not in nodes.columns or 'y' not in nodes.columns:
            return candidate
             
        for amenity, node_ids in candidate.placements.items():
            valid_nodes = []
            for nid in node_ids:
                nid = str(nid)
                if nid in nodes.index:
                    valid_nodes.append({"id": nid, "x": float(nodes.loc[nid, "x"]), "y": float(nodes.loc[nid, "y"])})
            
            accepted = []
            for node in valid_nodes:
                conflict = False
                for acc in accepted:
                    if math.sqrt((node["x"] - acc["x"])**2 + (node["y"] - acc["y"])**2) < min_spacing:
                        conflict = True
                        break
                if not conflict:
                    accepted.append(node)
            placements[amenity] = tuple(sorted(n["id"] for n in accepted))
        return Candidate(placements=placements, template_id=candidate.template_id)

    def persist_best(self, generation: int, candidate: Candidate, metrics: Dict[str, object]) -> None:
        cleaned_candidate = self.filter_conflicts(candidate)
        
        if self.landuse_filter is not None:
            try:
                placements_dict = dict(cleaned_candidate.placements)
                filtered_placements = self.landuse_filter.filter_candidate_placements(placements_dict)
                cleaned_candidate = Candidate(placements={k: tuple(v) for k, v in filtered_placements.items()}, template_id=cleaned_candidate.template_id)
            except Exception as e:
                logging.warning("Landuse filtering failed: %s", e)
        
        count_metrics = metrics.copy()
        if "placements" in count_metrics:
            count_metrics["placements"] = {k: len(v) for k, v in cleaned_candidate.placements.items()}
        
        payload = {"generation": generation, "candidate": cleaned_candidate.signature, "template": cleaned_candidate.template_id, "metrics": count_metrics}
        (self.analysis_dir / "best_candidate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI orchestration (config-driven)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid GA optimiser for amenity placement.")
    parser.add_argument("--city", type=str, default="bangalore", help="City to process")
    parser.add_argument("--mode", type=str, default="ga_only", choices=["ga_only", "ga_milp", "ga_milp_pnmlr"], help="Optimization mode")
    parser.add_argument("--verbosity", type=int, default=0, help="Logging verbosity (0=INFO, 1+=DEBUG)")
    return parser.parse_args()


def build_config(cfg: dict, analysis_dir: Path) -> HybridGAConfig:
    """Build HybridGAConfig entirely from YAML config."""
    opt = cfg.get('optimization', {})
    return HybridGAConfig(
        population=opt.get('ga_population', 80),
        generations=opt.get('ga_generations', 50),
        crossover_rate=opt.get('ga_crossover', 0.75),
        mutation_rate=opt.get('ga_mutation', 0.2),
        elitism=opt.get('ga_elitism', 4),
        seed_fraction=opt.get('ga_seed_fraction', 0.4),
        templates=opt.get('ga_templates', 5),
        local_search_budget=opt.get('ga_local_search_budget', 20),
        local_search_topk=opt.get('ga_local_search_topk', 5),
        per_pool_limit=opt.get('per_pool_limit', 500),
        random_seed=opt.get('ga_random_seed', 42),
        analysis_dir=analysis_dir,
        enable_progress=not opt.get('no_progress', False),
        workers=opt.get('ga_workers') or max(1, int(os.cpu_count() * 0.5)) if os.cpu_count() else 4,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbosity)
    
    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    cfg = cdm.load_config()
    
    opt_cfg = cfg.get('optimization', {})
    if opt_cfg.get('skip_ga', False):
        logging.info("skip_ga is True; skipping GA execution.")
        return
    
    nodes_path = cdm.baseline_nodes
    high_travel_path = cdm.high_travel_nodes(args.mode)
    analysis_dir = cdm.optimized_dir(args.mode)
    
    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes_with_scores parquet not found: {nodes_path}")
    if not high_travel_path.exists():
        raise FileNotFoundError(f"High travel node CSV not found: {high_travel_path}")

    nodes = ensure_index_on_osmid(pd.read_parquet(nodes_path))
    high_travel_nodes = ensure_index_on_osmid(pd.read_csv(high_travel_path))

    config = build_config(cfg, analysis_dir)
    config.analysis_dir.mkdir(parents=True, exist_ok=True)

    amenity_weights_cfg = cfg.get("amenity_weights", {})
    distance_columns = detect_distance_columns(nodes)
    amenity_weights = detect_amenity_weights(nodes, amenity_weights_cfg)
    min_spacing = cfg.get("amenity_distance_cutoff_m", 1200)
    amenity_pools = build_amenity_pools(high_travel_nodes, nodes, distance_columns, amenity_weights, config.per_pool_limit, min_spacing=min_spacing)
    
    amenity_budgets = {}
    milp_config = cfg.get("milp", {})
    if isinstance(milp_config, dict):
        budgets = milp_config.get("amenity_budgets", {})
        if isinstance(budgets, dict):
            amenity_budgets = {k: int(v) for k, v in budgets.items()}
    for amenity in amenity_weights.keys():
        if amenity not in amenity_budgets:
            amenity_budgets[amenity] = 20

    graph_path = cdm.processed_graph
    G = None
    if graph_path.exists():
        try:
            G = nx.read_graphml(str(graph_path))
            for u, v, k, data in G.edges(keys=True, data=True):
                if 'length' in data and isinstance(data['length'], str):
                    data['length'] = float(data['length'])
            logging.info("Loaded graph with %d nodes, %d edges", len(G.nodes), len(G.edges))
        except Exception as e:
            logging.warning("Failed to load graph: %s", e)

    context = GAContext(
        nodes=nodes,
        high_travel_nodes=high_travel_nodes,
        amenity_weights=amenity_weights,
        amenity_pools=amenity_pools,
        distance_columns=distance_columns,
        config=config,
        amenity_budgets=amenity_budgets,
        cdm=cdm,
        raw_config=cfg,
    )
    context.extra["graph"] = G

    analyse_distances(high_travel_nodes, context)

    # Check for PNMLR hooks
    precompute_hook = default_precompute_effects
    evaluate_hook = default_evaluate_candidate
    
    pnmlr_cfg = cfg.get("pnmlr", {})
    if isinstance(pnmlr_cfg, dict) and pnmlr_cfg.get("enabled", False):
        try:
            from pnmlr_hooks import pnmlr_precompute_hook, pnmlr_evaluate_hook
            precompute_hook = pnmlr_precompute_hook
            evaluate_hook = pnmlr_evaluate_hook
            logging.info("PNMLR hooks loaded successfully")
        except ImportError as e:
            logging.warning("Failed to import PNMLR hooks: %s", e)

    enable_landuse = cfg.get("optimization", {}).get("enable_landuse_filter", True)
    
    ga = HybridGA(
        context=context,
        config=config,
        precompute_hook=precompute_hook,
        evaluate_hook=evaluate_hook,
        enable_landuse_filter=enable_landuse,
        cdm=cdm,
        raw_config=cfg,
    )
    result = ga.run()
    logging.info("Best candidate: %s", result["best_candidate"])


if __name__ == "__main__":
    main()
