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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml
import networkx as nx

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
    population: int = 80
    generations: int = 50
    crossover_rate: float = 0.75
    mutation_rate: float = 0.2
    elitism: int = 4
    seed_fraction: float = 0.4
    templates: int = 5
    local_search_budget: int = 20
    local_search_topk: int = 5
    per_pool_limit: int = 200
    random_seed: int = 42
    analysis_dir: Path = Path("optimization") / "runs"
    # progress options
    enable_progress: bool = True
    progress_refresh: float = 0.5  # seconds between refreshes for non-tqdm fallback
    workers: int = 1


@dataclass
class GAContext:
    nodes: pd.DataFrame
    high_travel_nodes: pd.DataFrame
    amenity_weights: Dict[str, float]
    amenity_pools: Dict[str, List[str]]
    distance_columns: Dict[str, str]
    config: HybridGAConfig
    extra: Dict[str, object] = field(default_factory=dict)


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
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_yaml_config(config_path: Path) -> Dict[str, object]:
    if not config_path.exists():
        logging.warning("Configuration file %s missing; falling back to defaults.", config_path)
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    logging.info("Loaded configuration from %s", config_path)
    return data


def detect_amenity_weights(nodes: pd.DataFrame, cfg: Mapping[str, float], fallback: float = 1.0) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    prefix = "dist_to_"
    for column in nodes.columns:
        if not column.startswith(prefix):
            continue
        amenity = column[len(prefix) :].strip()
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
            mapping[column[len(prefix) :]] = column
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
            logging.debug("Distance column %s missing for amenity %s.", column, amenity)
            continue
        series = pd.to_numeric(common[column], errors="coerce")
        mean_dist = float(series.dropna().mean()) if not series.dropna().empty else float("inf")
        logging.info(
            "Amenity %-20s | mean distance %.2f m | available %4d",
            amenity,
            mean_dist,
            series.dropna().shape[0],
        )


def build_amenity_pools(
    high_nodes: pd.DataFrame,
    nodes: pd.DataFrame,
    distance_columns: Mapping[str, str],
    weights: Mapping[str, float],
    per_pool_limit: int,
) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    base_candidates = ensure_index_on_osmid(high_nodes)
    indexed_nodes = ensure_index_on_osmid(nodes)
    
    # Calculate city center for centrality bias
    center_x = indexed_nodes['x'].median()
    center_y = indexed_nodes['y'].median()
    
    for amenity in weights.keys():
        column = distance_columns.get(amenity)
        if column and column in indexed_nodes.columns:
            # Start with high travel nodes
            candidate_frame = indexed_nodes.loc[indexed_nodes.index.intersection(base_candidates.index)]
            
            # CRITICAL FIX: Exclude nodes already well-served by existing amenities
            # Minimum spacing: 1.5x the coverage threshold to ensure new placements add real value
            # For 1200m threshold → exclude nodes within 1800m of existing amenities
            min_spacing = 1200  # Conservative minimum: don't place where existing amenity is closer than this
            
            if column in candidate_frame.columns:
                # Filter out nodes with existing amenity too close
                before_count = len(candidate_frame)
                candidate_frame = candidate_frame[
                    (pd.to_numeric(candidate_frame[column], errors='coerce') >= min_spacing) |
                    (pd.to_numeric(candidate_frame[column], errors='coerce').isna())
                ]
                after_count = len(candidate_frame)
                if before_count > after_count:
                    logging.info(
                        "Filtered %d nodes for %s (too close to existing, < %dm)",
                        before_count - after_count, amenity, min_spacing
                    )
            
            # Calculate composite score: balance between distance from existing amenities and centrality
            if 'x' in candidate_frame.columns and 'y' in candidate_frame.columns:
                # Distance from city center (lower is better for central placement)
                candidate_frame = candidate_frame.copy()
                candidate_frame['dist_from_center'] = candidate_frame.apply(
                    lambda row: math.sqrt((row['x'] - center_x)**2 + (row['y'] - center_y)**2),
                    axis=1
                )
                
                # Normalize both metrics to 0-1 range
                existing_dist = pd.to_numeric(candidate_frame[column], errors='coerce').fillna(0)
                center_dist = candidate_frame['dist_from_center']
                
                # Normalize
                existing_dist_norm = (existing_dist - existing_dist.min()) / (existing_dist.max() - existing_dist.min() + 1)
                center_dist_norm = (center_dist - center_dist.min()) / (center_dist.max() - center_dist.min() + 1)
                
                # Composite score: 60% weight on distance from existing, 40% weight on centrality (lower center distance = better)
                candidate_frame['placement_score'] = 0.6 * existing_dist_norm - 0.4 * center_dist_norm
                
                candidates = candidate_frame.sort_values('placement_score', ascending=False)
            else:
                candidates = candidate_frame.sort_values(column, ascending=False)
        else:
            candidates = base_candidates.sort_values("travel_time_min", ascending=False)
        
        node_ids = [str(idx) for idx in candidates.index.tolist()]
        
        # Fallback if we filtered everything out
        if not node_ids and column and column in indexed_nodes.columns:
            logging.warning("All candidates filtered for %s, using fallback nodes with min_spacing check", amenity)
            fallback = indexed_nodes[
                (pd.to_numeric(indexed_nodes[column], errors='coerce') >= min_spacing) |
                (pd.to_numeric(indexed_nodes[column], errors='coerce').isna())
            ].sort_values(column, ascending=False)
            node_ids = [str(idx) for idx in fallback.index.tolist()]
        
        if per_pool_limit > 0:
            node_ids = node_ids[:per_pool_limit]
        
        pools[amenity] = node_ids
        logging.info("Amenity %s pool prepared with %d candidates (filtered for min %dm from existing).", amenity, len(node_ids), min_spacing)
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
    
    # Load graph for diversity penalty computation
    graph = context.extra.get("graph")
    
    # Precompute undirected graph ONCE to avoid expensive to_undirected() calls in evaluation
    graph_undir = None
    if graph is not None:
        logging.info("Precomputing undirected graph for diversity penalty...")
        graph_undir = graph.to_undirected()
    
    return {"nodes": indexed_nodes, "graph": graph, "graph_undir": graph_undir}


def default_evaluate_candidate(candidate: Candidate, effects: Mapping[str, object], context: GAContext) -> Dict[str, object]:
    nodes: pd.DataFrame = effects["nodes"]  # type: ignore[assignment]
    amenity_weights = context.amenity_weights
    baselines = context.extra.get("distance_baselines", {}) if isinstance(context.extra, dict) else {}
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
        baseline_distance = None
        if isinstance(baselines, Mapping):
            baseline_distance = baselines.get(amenity)
        if baseline_distance is None or not isinstance(baseline_distance, (int, float)) or not math.isfinite(baseline_distance):
            baseline_distance = float(numeric_distances.median()) if not numeric_distances.empty else 1.0
        if baseline_distance <= 0:
            baseline_distance = 1.0
        score = weight * (baseline_distance / (min_distance + 1.0))
        score = max(score, 0.0)
        amenity_scores[amenity] = score
        total_gain += score
        placements[amenity] = len(target_nodes)

    travel_penalty = 0.0
    if "travel_time_min" in nodes.columns:
        placed_index = [str(node) for ids in candidate.placements.values() for node in ids]
        placed_nodes = nodes.loc[nodes.index.isin(placed_index)]
        if not placed_nodes.empty:
            travel_penalty = float(placed_nodes["travel_time_min"].mean())

    # Add diversity penalty: penalize amenities of the same type placed too close together
    diversity_penalty = 0.0
    proximity_penalty = 0.0  # Penalty for placing too close to EXISTING amenities
    
    # Check proximity to existing amenities (fast - just column lookups)
    for amenity, node_ids in candidate.placements.items():
        column = context.distance_columns.get(amenity)
        if not column or column not in nodes.columns:
            continue
        
        for node in node_ids:
            node_str = str(node)
            if node_str in nodes.index:
                existing_dist = pd.to_numeric(nodes.loc[node_str, column], errors='coerce')
                if pd.notna(existing_dist) and existing_dist < 1200:
                    # Harsh penalty for placing within coverage threshold of existing amenity
                    # Penalty scales: 100% penalty at 0m, 0% penalty at 1200m
                    ratio = existing_dist / 1200.0
                    penalty = (1.0 - ratio) ** 2  # Exponential penalty
                    proximity_penalty += penalty * 2.0  # Strong penalty weight
    
    # Add Euclidean-based diversity penalty for new placements (FAST - no network ops)
    for amenity, node_ids in candidate.placements.items():
        if len(node_ids) <= 1:
            continue
        
        # Get coordinates for all placed nodes of this amenity type
        coords = []
        for node in node_ids:
            node_str = str(node)
            if node_str in nodes.index and 'x' in nodes.columns and 'y' in nodes.columns:
                x = nodes.loc[node_str, 'x']
                y = nodes.loc[node_str, 'y']
                if pd.notna(x) and pd.notna(y):
                    coords.append((float(x), float(y)))
        
        # Compute pairwise Euclidean distances (coords are already in meters - UTM projection)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist_meters = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # Strong penalty if new placements are too close to each other
                # Target: minimum 3000m spacing between new amenities of same type
                min_spacing = 3000.0
                if dist_meters < min_spacing:
                    ratio = dist_meters / min_spacing
                    penalty = (1.0 - ratio) ** 2  # Quadratic penalty
                    diversity_penalty += penalty * 5.0  # STRONG penalty weight

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

    # ------------------------------------------------------------------
    # Population seeding
    # ------------------------------------------------------------------

    def seed_population(self) -> List[Candidate]:
        population: List[Candidate] = []
        template_candidates = self.build_greedy_templates(self.config.templates)
        num_seeded = max(1, int(self.config.population * self.config.seed_fraction))
        seeded_templates = template_candidates[:num_seeded]
        population.extend(seeded_templates)
        logging.info("Seeded population with %d greedy templates.", len(seeded_templates))
        while len(population) < self.config.population:
            population.append(self.random_candidate())
        self.random.shuffle(population)
        return population

    def build_greedy_templates(self, max_templates: int) -> List[Candidate]:
        templates: List[Candidate] = []
        sorted_amenities = sorted(
            self.context.amenity_weights.items(), key=lambda item: item[1], reverse=True
        )
        pools = self.context.amenity_pools
        for template_idx in range(max_templates):
            placements: Dict[str, Tuple[str, ...]] = {}
            for amenity, _ in sorted_amenities:
                pool = pools.get(amenity, [])
                if not pool:
                    continue
                # Place 3-7 amenities per type (varied by template)
                num_to_place = 3 + template_idx % 5  # 3, 4, 5, 6, 7
                slice_start = min(template_idx * 2, max(0, len(pool) - num_to_place))
                slice_end = min(slice_start + num_to_place, len(pool))
                chosen = pool[slice_start:slice_end]
                placements[amenity] = tuple(chosen)
            templates.append(Candidate(placements=placements, template_id=f"greedy_{template_idx}"))
        return templates

    def random_candidate(self) -> Candidate:
        placements: Dict[str, Tuple[str, ...]] = {}
        for amenity, pool in self.context.amenity_pools.items():
            if not pool:
                continue
            # Sample 10-15% of pool, minimum 5, maximum 20
            sample_pct = 0.10 + (self.random.random() * 0.05)  # 10-15%
            sample_size = min(20, max(5, int(len(pool) * sample_pct)))
            sample_size = min(sample_size, len(pool))
            chosen = self.random.sample(pool, k=sample_size)
            placements[amenity] = tuple(sorted(chosen))
        return Candidate(placements=placements)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

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
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to load checkpoint %s: %s", path, exc)
            return None

    def _save_checkpoint(
        self,
        next_generation: int,
        population: Sequence[Candidate],
        history: Sequence[Dict[str, object]],
        best_overall: Optional[Tuple[Candidate, Dict[str, object]]],
    ) -> None:
        payload: Dict[str, object] = {
            "version": 1,
            "timestamp": time.time(),
            "next_generation": int(next_generation),
            "population": [serialize_candidate(candidate) for candidate in population],
            "history": list(history),
            "operator_credits": dict(self.operator_credits),
            "random_state": self.random.getstate(),
        }
        if best_overall:
            best_candidate, best_metrics = best_overall
            payload["best_overall"] = {
                "candidate": serialize_candidate(best_candidate),
                "metrics": best_metrics,
            }
        try:
            self.checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to persist checkpoint %s: %s", self.checkpoint_path, exc)

    def _clear_checkpoint(self) -> None:
        path = self.checkpoint_path
        if not path.exists():
            return
        try:
            path.unlink()
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug("Failed to remove checkpoint %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def crossover(self, a: Candidate, b: Candidate) -> Tuple[Candidate, Candidate]:
        if self.random.random() >= self.config.crossover_rate:
            return a, b
        child_a: Dict[str, Tuple[str, ...]] = {}
        child_b: Dict[str, Tuple[str, ...]] = {}
        amenities = set(a.placements.keys()) | set(b.placements.keys())
        for amenity in amenities:
            nodes_a = set(a.placements.get(amenity, ()))
            nodes_b = set(b.placements.get(amenity, ()))
            pool = self.context.amenity_pools.get(amenity, [])
            union = list(nodes_a | nodes_b)
            if not union:
                continue
            self.random.shuffle(union)
            pivot = self.random.randint(0, len(union))
            child_a_nodes = tuple(sorted(union[:pivot]))
            child_b_nodes = tuple(sorted(union[pivot:]))
            child_a[amenity] = child_a_nodes
            child_b[amenity] = child_b_nodes
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
                # remove one
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

    # ------------------------------------------------------------------
    # Evaluation and local search
    # ------------------------------------------------------------------

    def evaluate(self, candidate: Candidate) -> Dict[str, object]:
        return self.evaluate_hook(candidate, self.effects, self.context)

    def _evaluate_population_sequential(
        self,
        population: Sequence[Candidate],
        generation: int,
        sink: List[Tuple[Candidate, Dict[str, object]]],
    ) -> None:
        progress = None
        iterator: Iterable[Candidate]
        if _HAS_TQDM and self.config.enable_progress:
            progress = tqdm(population, desc=f"Gen {generation} eval", leave=False)
            iterator = progress
        else:
            iterator = population
        try:
            for candidate in iterator:
                metrics = self.evaluate(candidate)
                sink.append((candidate, metrics))
        finally:
            if progress is not None:
                progress.close()

    def _evaluate_population_parallel(
        self,
        population: Sequence[Candidate],
        generation: int,
        sink: List[Tuple[Candidate, Dict[str, object]]],
    ) -> None:
        total = len(population)
        if total == 0:
            return
        progress = None
        if _HAS_TQDM and self.config.enable_progress:
            progress = tqdm(total=total, desc=f"Gen {generation} eval", leave=False)
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {executor.submit(self.evaluate, candidate): candidate for candidate in population}
            try:
                for future in as_completed(futures):
                    candidate = futures[future]
                    metrics = future.result()
                    sink.append((candidate, metrics))
                    if progress is not None:
                        progress.update(1)
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                raise
            finally:
                if progress is not None:
                    progress.close()

    def local_search(self, candidate: Candidate, base_metrics: Dict[str, object]) -> Tuple[Candidate, Dict[str, object]]:
        best_candidate = candidate
        best_metrics = base_metrics
        best_fitness = float(base_metrics.get("fitness", float("-inf")))
        budget = self.config.local_search_budget
        if budget <= 0:
            return best_candidate, best_metrics

        # Use tqdm for local search progress if available and enabled
        amenity_pools_items = list(self.context.amenity_pools.items())
        if _HAS_TQDM and self.config.enable_progress:
            outer_iter = range(budget)
        else:
            outer_iter = range(budget)

        for _ in outer_iter:
            improved = False
            # iterate amenities; break early if improved
            for amenity, pool in amenity_pools_items:
                current_nodes = set(best_candidate.placements.get(amenity, ()))
                
                # Parallel evaluation of neighbors when using multiple workers
                if self.config.workers > 1:
                    # Build candidate neighbors
                    neighbors = []
                    for node in pool[:min(50, len(pool))]:  # Limit to top 50 for performance
                        if node in current_nodes:
                            continue
                        trial_nodes = tuple(sorted(current_nodes | {node}))
                        trial_placements = dict(best_candidate.placements)
                        trial_placements[amenity] = trial_nodes
                        trial_candidate = Candidate(placements=trial_placements, template_id=best_candidate.template_id)
                        neighbors.append((node, trial_candidate))
                    
                    # Evaluate neighbors in parallel
                    if neighbors:
                        with ThreadPoolExecutor(max_workers=min(4, self.config.workers)) as executor:
                            futures = {executor.submit(self.evaluate, cand): (node, cand) for node, cand in neighbors}
                            for future in as_completed(futures):
                                node, trial_candidate = futures[future]
                                trial_metrics = future.result()
                                trial_fitness = float(trial_metrics.get("fitness", float("-inf")))
                                if trial_fitness > best_fitness:
                                    best_candidate = trial_candidate
                                    best_metrics = trial_metrics
                                    best_fitness = trial_fitness
                                    improved = True
                                    self.operator_credits["local_search"] += 1
                else:
                    # Sequential evaluation for single worker
                    for node in pool:
                        if node in current_nodes:
                            continue
                        trial_nodes = tuple(sorted(current_nodes | {node}))
                        trial_placements = dict(best_candidate.placements)
                        trial_placements[amenity] = trial_nodes
                        trial_candidate = Candidate(placements=trial_placements, template_id=best_candidate.template_id)
                        trial_metrics = self.evaluate(trial_candidate)
                        trial_fitness = float(trial_metrics.get("fitness", float("-inf")))
                        if trial_fitness > best_fitness:
                            best_candidate = trial_candidate
                            best_metrics = trial_metrics
                            best_fitness = trial_fitness
                            improved = True
                            self.operator_credits["local_search"] += 1
                            break
                
                if improved:
                    break
            if not improved:
                break
        return best_candidate, best_metrics

    # ------------------------------------------------------------------
    # GA loop
    # ------------------------------------------------------------------

    def _write_heartbeat(self, generation: int, best_metrics: Dict[str, object], elapsed: float) -> None:
        try:
            heartbeat = {
                "generation": generation,
                "best_fitness": float(best_metrics.get("fitness", 0.0)) if best_metrics else None,
                "operator_credits": dict(self.operator_credits),
                "elapsed_seconds": elapsed,
                "timestamp": time.time(),
            }
            path = self.analysis_dir / "heartbeat.json"
            path.write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")
        except Exception as e:
            logging.debug("Failed to write heartbeat: %s", e)

    def run(self) -> Dict[str, object]:
        logging.info("Starting hybrid GA: population=%d, generations=%d", self.config.population, self.config.generations)
        start_run = time.time()
        self.effects = self.precompute_hook(self.context)
        if self.config.workers > 1:
            logging.info("Using %d worker threads for candidate evaluation.", self.config.workers)
        resume_data = self._load_checkpoint()
        history: List[Dict[str, object]] = []
        best_overall: Optional[Tuple[Candidate, Dict[str, object]]] = None

        if resume_data:
            try:
                start_generation = int(resume_data.get("next_generation", 1))
            except (TypeError, ValueError):
                start_generation = 1
            start_generation = max(1, start_generation)

            population_payload = resume_data.get("population", [])
            population: List[Candidate] = []
            if isinstance(population_payload, list):
                for item in population_payload:
                    if isinstance(item, Mapping):
                        population.append(deserialize_candidate(item))

            history_payload = resume_data.get("history", [])
            if isinstance(history_payload, list):
                for entry in history_payload:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        generation_value = int(entry.get("generation", 0))
                    except (TypeError, ValueError):
                        generation_value = 0
                    if generation_value < start_generation:
                        history.append(entry)

            best_payload = resume_data.get("best_overall")
            if isinstance(best_payload, Mapping):
                candidate_payload = best_payload.get("candidate")
                metrics_payload = best_payload.get("metrics", {})
                if isinstance(candidate_payload, Mapping):
                    metrics_dict = metrics_payload if isinstance(metrics_payload, dict) else {}
                    best_overall = (deserialize_candidate(candidate_payload), metrics_dict)

            random_state_payload = resume_data.get("random_state")
            if random_state_payload is not None:
                try:
                    self.random.setstate(_lists_to_tuples(random_state_payload))
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("Failed to restore RNG state from checkpoint: %s", exc)

            stored_credits = resume_data.get("operator_credits")
            if isinstance(stored_credits, Mapping):
                for key, value in stored_credits.items():
                    if key in self.operator_credits:
                        try:
                            self.operator_credits[key] = int(value)
                        except (TypeError, ValueError):
                            logging.debug("Ignoring invalid operator credit for %s", key)

            if self.config.generations >= 1:
                start_generation = min(start_generation, self.config.generations)
                if history:
                    filtered_history: List[Dict[str, object]] = []
                    for entry in history:
                        try:
                            gen_val = int(entry.get("generation", 0))
                        except (TypeError, ValueError):
                            gen_val = 0
                        if gen_val < start_generation:
                            filtered_history.append(entry)
                    history = filtered_history

            if not population:
                logging.info("Checkpoint missing usable population. Reseeding from scratch.")
                population = self.seed_population()
                history = []
                best_overall = None
                start_generation = 1
            else:
                logging.info("Resuming from generation %d with %d individuals.", start_generation, len(population))
        else:
            population = self.seed_population()
            history = []
            best_overall = None
            start_generation = 1

        gen_range = range(start_generation, self.config.generations + 1)
        use_tqdm_gen = _HAS_TQDM and self.config.enable_progress
        gen_iterator = tqdm(gen_range, desc="Generations", unit="gen") if use_tqdm_gen else gen_range

        current_generation = start_generation
        self._save_checkpoint(start_generation, population, history, best_overall)

        try:
            for generation in gen_iterator:
                current_generation = generation
                gen_start = time.time()
                evaluated: List[Tuple[Candidate, Dict[str, object]]] = []
                logging.info("Evaluating generation %d/%d", generation, self.config.generations)

                try:
                    if self.config.workers > 1:
                        self._evaluate_population_parallel(population, generation, evaluated)
                    else:
                        self._evaluate_population_sequential(population, generation, evaluated)
                except KeyboardInterrupt:
                    if evaluated:
                        evaluated.sort(key=lambda item: float(item[1].get("fitness", float("-inf"))), reverse=True)
                        best_candidate, best_metrics = evaluated[0]
                        elapsed = time.time() - gen_start
                        logging.info(
                            "Interrupted during generation %d after evaluating %d candidates. Current best fitness %.4f.",
                            generation,
                            len(evaluated),
                            float(best_metrics.get("fitness", 0.0)),
                        )
                        self.persist_generation(generation, evaluated[:10])
                        if best_overall is None or float(best_metrics.get("fitness", float("-inf"))) > float(best_overall[1].get("fitness", float("-inf"))):
                            best_overall = (best_candidate, best_metrics)
                            self.persist_best(generation, best_candidate, best_metrics)
                        self._write_heartbeat(generation, best_metrics, elapsed)
                    self._save_checkpoint(generation, population, history, best_overall)
                    raise

                if not evaluated:
                    logging.warning("Generation %d produced no evaluations; skipping.", generation)
                    self._save_checkpoint(generation + 1, population, history, best_overall)
                    continue

                # Sort & local-search top-k — show progress for local search
                evaluated.sort(key=lambda item: float(item[1].get("fitness", float("-inf"))), reverse=True)
                top_candidates = evaluated[: self.config.local_search_topk]
                enhanced: List[Tuple[Candidate, Dict[str, object]]] = []
                if _HAS_TQDM and self.config.enable_progress:
                    ls_iter = tqdm(top_candidates, desc=f"Gen {generation} local_search", leave=False)
                else:
                    ls_iter = top_candidates
                try:
                    for candidate, metrics in ls_iter:
                        enhanced_candidate, enhanced_metrics = self.local_search(candidate, metrics)
                        enhanced.append((enhanced_candidate, enhanced_metrics))
                except KeyboardInterrupt:
                    if evaluated:
                        evaluated.sort(key=lambda item: float(item[1].get("fitness", float("-inf"))), reverse=True)
                        best_candidate, best_metrics = evaluated[0]
                        elapsed = time.time() - gen_start
                        logging.info(
                            "Interrupted during local search in generation %d. Current best fitness %.4f.",
                            generation,
                            float(best_metrics.get("fitness", 0.0)),
                        )
                        self.persist_generation(generation, evaluated[:10])
                        if best_overall is None or float(best_metrics.get("fitness", float("-inf"))) > float(best_overall[1].get("fitness", float("-inf"))):
                            best_overall = (best_candidate, best_metrics)
                            self.persist_best(generation, best_candidate, best_metrics)
                        self._write_heartbeat(generation, best_metrics, elapsed)
                    self._save_checkpoint(generation, population, history, best_overall)
                    raise
                evaluated[: self.config.local_search_topk] = enhanced
                evaluated.sort(key=lambda item: float(item[1].get("fitness", float("-inf"))), reverse=True)

                best_candidate, best_metrics = evaluated[0]
                gen_elapsed = time.time() - gen_start
                logging.info(
                    "Generation %d best fitness %.4f | template=%s | placements=%s | elapsed=%.2fs",
                    generation,
                    float(best_metrics.get("fitness", 0.0)),
                    best_candidate.template_id,
                    best_metrics.get("placements"),
                    gen_elapsed,
                )

                # persist best & heartbeat
                if best_overall is None or float(best_metrics.get("fitness", float("-inf"))) > float(best_overall[1].get("fitness", float("-inf"))):
                    best_overall = (best_candidate, best_metrics)
                    self.persist_best(generation, best_candidate, best_metrics)
                self._write_heartbeat(generation, best_metrics, gen_elapsed)

                history.append(
                    {
                        "generation": generation,
                        "best_fitness": float(best_metrics.get("fitness", 0.0)),
                        "operator_credits": dict(self.operator_credits),
                    }
                )
                self.persist_generation(generation, evaluated[:10])

                # Build next population
                elites = [item[0] for item in evaluated[: self.config.elitism]]
                next_population: List[Candidate] = elites.copy()

                while len(next_population) < self.config.population:
                    parent_a = self.tournament_select(evaluated)
                    parent_b = self.tournament_select(evaluated)
                    child_a, child_b = self.crossover(parent_a, parent_b)
                    child_a = self.mutate(child_a)
                    next_population.append(child_a)
                    if len(next_population) < self.config.population:
                        child_b = self.mutate(child_b)
                        next_population.append(child_b)

                self._save_checkpoint(generation + 1, next_population, history, best_overall)
                population = next_population
        except KeyboardInterrupt:
            logging.warning("Interrupted at generation %d; checkpoint saved to %s", current_generation, self.checkpoint_path)
            self._save_checkpoint(current_generation, population, history, best_overall)
            raise

        assert best_overall is not None
        result = {
            "best_candidate": best_overall[0].signature,
            "best_metrics": best_overall[1],
            "history": history,
            "operator_credits": dict(self.operator_credits),
        }
        summary_path = self.analysis_dir / "summary.json"
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        total_elapsed = time.time() - start_run
        logging.info("GA complete. Summary written to %s | total elapsed %.2fs", summary_path, total_elapsed)
        self._clear_checkpoint()
        return result

    def tournament_select(self, evaluated: Sequence[Tuple[Candidate, Dict[str, object]]], k: int = 3) -> Candidate:
        participants = self.random.sample(evaluated, k=min(k, len(evaluated)))
        participants.sort(key=lambda item: float(item[1].get("fitness", float("-inf"))), reverse=True)
        return participants[0][0]

    def persist_generation(self, generation: int, top_entries: Sequence[Tuple[Candidate, Dict[str, object]]]) -> None:
        payload = [
            {
                "candidate": candidate.signature,
                "metrics": metrics,
            }
            for candidate, metrics in top_entries
        ]
        path = self.analysis_dir / f"generation_{generation:04d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def persist_best(self, generation: int, candidate: Candidate, metrics: Dict[str, object]) -> None:
        path = self.analysis_dir / "best_candidate.json"
        payload = {
            "generation": generation,
            "candidate": candidate.signature,
            "template": candidate.template_id,
            "metrics": metrics,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Hybrid GA optimiser for amenity placement.")
    parser.add_argument("--nodes-scores", type=Path, default=project_root / "data" / "analysis" / "nodes_with_scores.parquet")
    parser.add_argument("--high-travel", type=Path, default=project_root / "optimization" / "high_travel_time_nodes.csv")
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--crossover", type=float, default=0.75)
    parser.add_argument("--mutation", type=float, default=0.2)
    parser.add_argument("--elitism", type=int, default=4)
    parser.add_argument("--seed-fraction", type=float, default=0.4)
    parser.add_argument("--templates", type=int, default=5)
    parser.add_argument("--local-search-budget", type=int, default=30)
    parser.add_argument("--local-search-topk", type=int, default=5)
    parser.add_argument("--per-pool-limit", type=int, default=200)
    parser.add_argument("--analysis-dir", type=Path, default=project_root / "optimization" / "runs")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.cpu_count() * 0.5)) if os.cpu_count() else 4,
        help="Number of worker threads used to evaluate each generation in parallel. Default: 50%% of CPU cores.",
    )
    parser.add_argument("--precompute-hook", type=str, default=None)
    parser.add_argument("--evaluate-hook", type=str, default=None)
    parser.add_argument("--verbosity", type=int, default=0, help="Increase for debug logging.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bars (useful on non-interactive terminals).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> HybridGAConfig:
    return HybridGAConfig(
        population=args.population,
        generations=args.generations,
        crossover_rate=args.crossover,
        mutation_rate=args.mutation,
        elitism=args.elitism,
        seed_fraction=args.seed_fraction,
        templates=args.templates,
        local_search_budget=args.local_search_budget,
        local_search_topk=args.local_search_topk,
        per_pool_limit=args.per_pool_limit,
        random_seed=args.random_seed,
        analysis_dir=args.analysis_dir,
        enable_progress=not args.no_progress,
        workers=max(1, args.workers),
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbosity)
    config = build_config(args)

    nodes_path: Path = args.nodes_scores
    high_travel_path: Path = args.high_travel
    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes_with_scores parquet not found: {nodes_path}")
    if not high_travel_path.exists():
        raise FileNotFoundError(f"High travel node CSV not found: {high_travel_path}")

    nodes = ensure_index_on_osmid(pd.read_parquet(nodes_path))
    high_travel_nodes = ensure_index_on_osmid(pd.read_csv(high_travel_path))

    cfg_yaml = load_yaml_config(args.config)
    config.analysis_dir.mkdir(parents=True, exist_ok=True)

    amenity_weights_cfg = cfg_yaml.get("amenity_weights", {}) if isinstance(cfg_yaml, dict) else {}
    distance_columns = detect_distance_columns(nodes)
    amenity_weights = detect_amenity_weights(nodes, amenity_weights_cfg)
    amenity_pools = build_amenity_pools(high_travel_nodes, nodes, distance_columns, amenity_weights, config.per_pool_limit)

    # Load graph for diversity penalty computation
    logging.info("Loading graph for diversity penalty computation...")
    graph_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "graph.graphml"
    G = None
    if graph_path.exists():
        try:
            import networkx as nx
            G = nx.read_graphml(str(graph_path))
            # Fix edge weights
            for u, v, k, data in G.edges(keys=True, data=True):
                if 'length' in data and isinstance(data['length'], str):
                    data['length'] = float(data['length'])
            logging.info("Loaded graph with %d nodes, %d edges for diversity penalty", len(G.nodes), len(G.edges))
        except Exception as e:
            logging.warning("Failed to load graph: %s (diversity penalty disabled)", e)
    else:
        logging.warning("Graph not found at %s (diversity penalty disabled)", graph_path)

    context = GAContext(
        nodes=nodes,
        high_travel_nodes=high_travel_nodes,
        amenity_weights=amenity_weights,
        amenity_pools=amenity_pools,
        distance_columns=distance_columns,
        config=config,
    )
    
    # Add graph to context for diversity penalty
    context.extra["graph"] = G

    analyse_distances(high_travel_nodes, context)

    precompute_hook = load_callable(args.precompute_hook) or default_precompute_effects
    evaluate_hook = load_callable(args.evaluate_hook) or default_evaluate_candidate

    ga = HybridGA(context=context, config=config, precompute_hook=precompute_hook, evaluate_hook=evaluate_hook)
    result = ga.run()
    logging.info("Best candidate: %s", result["best_candidate"])


if __name__ == "__main__":
    main()
