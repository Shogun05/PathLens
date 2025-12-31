"""
Hybrid GA-MILP Refinement Module

Provides local MILP-based refinement for genetic algorithm candidates.
Operates on small optimization windows to improve individual fitness
without disrupting global exploration.

Design Philosophy:
    - GA: Global exploration across city
    - MILP: Local exploitation within neighborhoods
    - Hybrid: Combine strengths of both approaches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass, asdict
import time
import json
import hashlib
from pathlib import Path

try:
    import pulp
except ImportError:
    pulp = None

logger = logging.getLogger(__name__)


@dataclass
class MILPRefinementConfig:
    """Configuration for hybrid MILP refinement."""
    enabled: bool = True
    time_limit_seconds: float = 2.0  # Fast solve for GA integration
    max_amenities_to_relocate: int = 4  # Small window size
    max_hexagons_to_optimize: int = 3  # Focus on worst areas
    selection_strategy: str = "worst_hexagons"  # or "random", "least_contributing"
    apply_to_elite_only: bool = True  # Refine only top candidates
    apply_every_n_generations: int = 1  # Frequency
    min_improvement_threshold: float = 0.01  # Accept only if improves
    max_candidates_per_generation: int = 5  # Limit MILP calls
    enable_caching: bool = True  # Cache MILP solutions
    cache_dir: str = "../data/cache/milp"  # Cache directory


@dataclass
class RefinementResult:
    """Result of MILP refinement attempt."""
    success: bool
    improved: bool
    original_fitness: float
    refined_fitness: float
    improvement: float
    solve_time: float
    amenities_relocated: Dict[str, int]  # type -> count
    solver_status: str
    window_size: int  # Number of hexagons optimized
    cache_hit: bool = False  # Whether result came from cache


class HybridMILPRefiner:
    """
    MILP-based local refinement for GA candidates.
    
    Takes a GA individual (amenity placement configuration) and attempts
    to improve it by solving a small MILP over a localized optimization window.
    """
    
    def __init__(self, config: MILPRefinementConfig):
        """
        Initialize refiner.
        
        Args:
            config: MILP refinement configuration
        """
        if pulp is None:
            raise ImportError("PuLP required for MILP refinement. Install: pip install pulp")
        
        self.config = config
        self.stats = {
            'total_attempts': 0,
            'successful_solves': 0,
            'improvements': 0,
            'total_improvement': 0.0,
            'total_solve_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Setup cache directory
        if self.config.enable_caching:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"MILP refinement cache enabled: {self.cache_dir}")
        else:
            self.cache_dir = None
            logger.info("MILP refinement cache disabled")
    
    def should_refine_candidate(
        self,
        generation: int,
        candidate_rank: int,
        total_candidates: int
    ) -> bool:
        """
        Determine if a candidate should be refined based on policy.
        
        Args:
            generation: Current generation number
            candidate_rank: Rank in population (0 = best)
            total_candidates: Total population size
            
        Returns:
            True if should apply MILP refinement
        """
        if not self.config.enabled:
            return False
        
        # Check generation frequency
        if generation % self.config.apply_every_n_generations != 0:
            return False
        
        # Check if we've hit the per-generation limit
        if candidate_rank >= self.config.max_candidates_per_generation:
            return False
        
        # Check elite policy
        if self.config.apply_to_elite_only:
            # Only refine top 20% of population
            elite_threshold = max(1, int(0.2 * total_candidates))
            if candidate_rank >= elite_threshold:
                return False
        
        return True
    
    def _compute_candidate_hash(
        self,
        candidate: Dict[str, List[int]],
        window: Dict
    ) -> str:
        """
        Compute hash of candidate and window for caching.
        
        Args:
            candidate: Amenity placements
            window: Optimization window
            
        Returns:
            Hash string
        """
        # Create deterministic representation
        cache_key = {
            'candidate': {k: sorted(v) for k, v in candidate.items()},
            'window': {
                'hexagons': sorted(window.get('hexagons', [])),
                'amenities': {k: sorted(v) for k, v in window.get('amenities', {}).items()}
            }
        }
        
        # Hash the JSON representation
        key_str = json.dumps(cache_key, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_hash: str) -> Optional[Dict]:
        """Load refinement result from cache."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                logger.debug(f"Cache hit: {cache_hash[:8]}...")
                self.stats['cache_hits'] += 1
                return cached
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_hash[:8]}: {e}")
                return None
        
        self.stats['cache_misses'] += 1
        return None
    
    def _save_to_cache(
        self,
        cache_hash: str,
        refined_candidate: Dict[str, List[int]],
        result: RefinementResult
    ) -> None:
        """Save refinement result to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_hash}.json"
        try:
            cache_data = {
                'refined_candidate': refined_candidate,
                'result': asdict(result),
                'timestamp': time.time()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Cached refinement: {cache_hash[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_hash[:8]}: {e}")
    
    def refine_candidate(
        self,
        candidate: Dict[str, List[int]],
        nodes_df: pd.DataFrame,
        distance_cache: Dict,
        amenity_weights: Dict[str, float],
        distance_thresholds: Dict[str, float],
        candidate_pool: List[int],
        current_fitness: float
    ) -> Tuple[Dict[str, List[int]], RefinementResult]:
        """
        Apply MILP refinement to a GA candidate.
        
        Args:
            candidate: Current amenity placements {amenity_type: [node_ids]}
            nodes_df: DataFrame with node scores and H3 hexagons
            distance_cache: Precomputed distances {(node, amenity): distance}
            amenity_weights: Importance weights per amenity type
            distance_thresholds: Coverage thresholds per amenity type
            candidate_pool: Valid candidate node IDs for placement
            current_fitness: Current fitness value
            
        Returns:
            Tuple of (refined_candidate, refinement_result)
        """
        self.stats['total_attempts'] += 1
        start_time = time.time()
        
        logger.debug(f"Refining candidate (fitness={current_fitness:.4f})")
        
        try:
            # Step 1: Select optimization window
            window = self._select_optimization_window(
                candidate, nodes_df, distance_cache, amenity_weights
            )
            
            if not window['hexagons'] and not window['amenities']:
                logger.debug("Empty optimization window, skipping refinement")
                return candidate, RefinementResult(
                    success=False, improved=False, original_fitness=current_fitness,
                    refined_fitness=current_fitness, improvement=0.0,
                    solve_time=time.time() - start_time, amenities_relocated={},
                    solver_status="EmptyWindow", window_size=0
                )
            
            logger.debug(f"Window: {len(window.get('hexagons', []))} hexagons, "
                        f"{sum(len(v) for v in window['amenities'].values())} amenities")
            
            # Check cache
            cache_hash = None
            if self.config.enable_caching:
                cache_hash = self._compute_candidate_hash(candidate, window)
                cached = self._load_from_cache(cache_hash)
                if cached:
                    refined_candidate = cached['refined_candidate']
                    result_dict = cached['result']
                    result_dict['cache_hit'] = True
                    result = RefinementResult(**result_dict)
                    logger.debug(f"Returning cached refinement (improvement={result.improvement:.4f})")
                    
                    if result.improved:
                        return refined_candidate, result
                    else:
                        return candidate, result
            
            # Step 2: Build localized MILP
            logger.debug("Building local MILP...")
            problem, var_mapping = self._build_local_milp(
                candidate, window, nodes_df, distance_cache,
                amenity_weights, distance_thresholds, candidate_pool
            )
            
            # Step 3: Solve with time limit
            logger.debug(f"Solving MILP (time limit={self.config.time_limit_seconds}s)...")
            solver = pulp.PULP_CBC_CMD(
                timeLimit=self.config.time_limit_seconds,
                msg=False
            )
            status = problem.solve(solver)
            solve_time = time.time() - start_time
            
            status_str = pulp.LpStatus[status]
            self.stats['total_solve_time'] += solve_time
            
            logger.debug(f"Solver status: {status_str}, time: {solve_time:.2f}s")
            
            if status == pulp.LpStatusOptimal or status == pulp.LpStatusNotSolved:
                self.stats['successful_solves'] += 1
            
            # Step 4: Extract refined placements
            refined_candidate = self._extract_refined_candidate(
                candidate, problem, var_mapping, window
            )
            
            # Step 5: Evaluate fitness improvement
            refined_fitness = self._estimate_fitness_improvement(
                candidate, refined_candidate, window, nodes_df,
                distance_cache, amenity_weights
            )
            
            improvement = refined_fitness - current_fitness
            improved = improvement > self.config.min_improvement_threshold
            
            logger.debug(f"Fitness: {current_fitness:.4f} -> {refined_fitness:.4f} "
                        f"(Δ={improvement:.4f}, improved={improved})")
            
            if improved:
                self.stats['improvements'] += 1
                self.stats['total_improvement'] += improvement
            
            # Count relocated amenities
            relocated = self._count_relocations(candidate, refined_candidate)
            if relocated:
                logger.debug(f"Relocations: {relocated}")
            
            result = RefinementResult(
                success=True,
                improved=improved,
                original_fitness=current_fitness,
                refined_fitness=refined_fitness,
                improvement=improvement,
                solve_time=solve_time,
                amenities_relocated=relocated,
                solver_status=status_str,
                window_size=len(window['hexagons']),
                cache_hit=False
            )
            
            # Cache the result
            if cache_hash:
                self._save_to_cache(cache_hash, refined_candidate, result)
            
            # Return refined candidate only if improved
            if improved:
                logger.info(f"MILP refinement improved fitness by {improvement:.4f}")
                return refined_candidate, result
            else:
                logger.debug("MILP refinement did not improve fitness")
                return candidate, result
        
        except Exception as e:
            logger.warning(f"MILP refinement failed: {e}", exc_info=True)
            return candidate, RefinementResult(
                success=False, improved=False, original_fitness=current_fitness,
                refined_fitness=current_fitness, improvement=0.0,
                solve_time=time.time() - start_time, amenities_relocated={},
                solver_status=f"Error: {str(e)}", window_size=0
            )
    
    def _select_optimization_window(
        self,
        candidate: Dict[str, List[int]],
        nodes_df: pd.DataFrame,
        distance_cache: Dict,
        amenity_weights: Dict[str, float]
    ) -> Dict:
        """
        Select hexagons and amenities to optimize.
        
        Returns:
            Dict with 'hexagons': [hex_ids], 'amenities': {type: [nodes]}
        """
        if 'h3_08' not in nodes_df.columns:
            # Fallback: random selection
            return self._random_selection_window(candidate)
        
        if self.config.selection_strategy == "worst_hexagons":
            return self._select_worst_hexagons(
                candidate, nodes_df, distance_cache, amenity_weights
            )
        elif self.config.selection_strategy == "least_contributing":
            return self._select_least_contributing(
                candidate, nodes_df, distance_cache, amenity_weights
            )
        else:  # random
            return self._random_selection_window(candidate)
    
    def _select_worst_hexagons(
        self,
        candidate: Dict[str, List[int]],
        nodes_df: pd.DataFrame,
        distance_cache: Dict,
        amenity_weights: Dict[str, float]
    ) -> Dict:
        """Select hexagons with worst accessibility for optimization."""
        # Compute accessibility score per hexagon
        hex_scores = {}
        
        for hex_id in nodes_df['h3_08'].unique():
            hex_nodes = nodes_df[nodes_df['h3_08'] == hex_id]['osmid'].values
            
            if len(hex_nodes) == 0:
                continue
            
            # Compute mean accessibility for this hexagon
            total_accessibility = 0.0
            for node in hex_nodes:
                for amenity_type, placements in candidate.items():
                    weight = amenity_weights.get(amenity_type, 1.0)
                    
                    # Find nearest placed amenity
                    min_dist = float('inf')
                    for placed_node in placements:
                        dist = distance_cache.get((node, placed_node), float('inf'))
                        min_dist = min(min_dist, dist)
                    
                    # Inverse distance accessibility
                    if min_dist < float('inf'):
                        total_accessibility += weight / (min_dist + 1)
            
            hex_scores[hex_id] = total_accessibility / len(hex_nodes)
        
        # Select worst hexagons
        sorted_hexes = sorted(hex_scores.items(), key=lambda x: x[1])
        worst_hexes = [h[0] for h in sorted_hexes[:self.config.max_hexagons_to_optimize]]
        
        # Select amenities within these hexagons
        window_amenities = {}
        for amenity_type, placements in candidate.items():
            type_placements = []
            for node in placements:
                node_hex = nodes_df[nodes_df['osmid'] == node]['h3_08'].values
                if len(node_hex) > 0 and node_hex[0] in worst_hexes:
                    type_placements.append(node)
            
            if type_placements:
                window_amenities[amenity_type] = type_placements
        
        # Limit total amenities to relocate
        if sum(len(v) for v in window_amenities.values()) > self.config.max_amenities_to_relocate:
            window_amenities = self._trim_amenity_window(window_amenities)
        
        return {'hexagons': worst_hexes, 'amenities': window_amenities}
    
    def _select_least_contributing(
        self,
        candidate: Dict[str, List[int]],
        nodes_df: pd.DataFrame,
        distance_cache: Dict,
        amenity_weights: Dict[str, float]
    ) -> Dict:
        """Select amenities contributing least to overall accessibility."""
        # Compute contribution score per amenity placement
        contributions = []
        
        for amenity_type, placements in candidate.items():
            weight = amenity_weights.get(amenity_type, 1.0)
            
            for node in placements:
                # Count how many demand nodes this placement serves
                served_count = 0
                for demand_node in nodes_df['osmid'].values:
                    dist = distance_cache.get((demand_node, node), float('inf'))
                    if dist < 1000:  # Within reasonable distance
                        served_count += 1
                
                contribution = weight * served_count
                contributions.append((amenity_type, node, contribution))
        
        # Select least contributing amenities
        sorted_contrib = sorted(contributions, key=lambda x: x[2])
        to_relocate = sorted_contrib[:self.config.max_amenities_to_relocate]
        
        window_amenities = {}
        affected_hexes = set()
        
        for amenity_type, node, _ in to_relocate:
            if amenity_type not in window_amenities:
                window_amenities[amenity_type] = []
            window_amenities[amenity_type].append(node)
            
            # Track affected hexagons
            node_hex = nodes_df[nodes_df['osmid'] == node]['h3_08'].values
            if len(node_hex) > 0:
                affected_hexes.add(node_hex[0])
        
        return {'hexagons': list(affected_hexes), 'amenities': window_amenities}
    
    def _random_selection_window(self, candidate: Dict[str, List[int]]) -> Dict:
        """Randomly select amenities to relocate."""
        window_amenities = {}
        
        all_placements = []
        for amenity_type, placements in candidate.items():
            for node in placements:
                all_placements.append((amenity_type, node))
        
        if len(all_placements) == 0:
            return {'hexagons': [], 'amenities': {}}
        
        # Random sample
        n_to_select = min(self.config.max_amenities_to_relocate, len(all_placements))
        selected = np.random.choice(len(all_placements), n_to_select, replace=False)
        
        for idx in selected:
            amenity_type, node = all_placements[idx]
            if amenity_type not in window_amenities:
                window_amenities[amenity_type] = []
            window_amenities[amenity_type].append(node)
        
        return {'hexagons': [], 'amenities': window_amenities}
    
    def _trim_amenity_window(self, window_amenities: Dict[str, List[int]]) -> Dict:
        """Trim window to max_amenities_to_relocate."""
        trimmed = {}
        total = 0
        
        for amenity_type, placements in window_amenities.items():
            remaining = self.config.max_amenities_to_relocate - total
            if remaining <= 0:
                break
            
            to_include = placements[:remaining]
            if to_include:
                trimmed[amenity_type] = to_include
                total += len(to_include)
        
        return trimmed
    
    def _build_local_milp(
        self,
        candidate: Dict[str, List[int]],
        window: Dict,
        nodes_df: pd.DataFrame,
        distance_cache: Dict,
        amenity_weights: Dict[str, float],
        distance_thresholds: Dict[str, float],
        candidate_pool: List[int]
    ) -> Tuple[pulp.LpProblem, Dict]:
        """Build localized MILP for window optimization."""
        prob = pulp.LpProblem("LocalRefinement", pulp.LpMaximize)
        
        # Filter demand nodes to window hexagons (if available)
        if window['hexagons'] and 'h3_08' in nodes_df.columns:
            demand_nodes = nodes_df[
                nodes_df['h3_08'].isin(window['hexagons'])
            ]['osmid'].values
        else:
            # Use all nodes if no hexagon filtering
            demand_nodes = nodes_df['osmid'].values
        
        # Limit demand nodes for speed
        if len(demand_nodes) > 200:
            demand_nodes = np.random.choice(demand_nodes, 200, replace=False)
        
        # Filter candidate pool to reasonable locations
        local_candidates = [
            c for c in candidate_pool
            if any(distance_cache.get((c, d), float('inf')) < 5000 for d in demand_nodes)
        ][:50]  # Limit for speed
        
        # Decision variables: x[i,a] for amenities in window
        x = {}
        for amenity_type, nodes_to_relocate in window['amenities'].items():
            for candidate_node in local_candidates:
                x[(candidate_node, amenity_type)] = pulp.LpVariable(
                    f"place_{candidate_node}_{amenity_type}",
                    cat=pulp.LpBinary
                )
        
        # Coverage variables
        y = {}
        for node in demand_nodes:
            for amenity_type in window['amenities'].keys():
                y[(node, amenity_type)] = pulp.LpVariable(
                    f"cover_{node}_{amenity_type}",
                    cat=pulp.LpBinary
                )
        
        # Objective: maximize local accessibility improvement
        node_weights = nodes_df.set_index('osmid')['population_weight'].to_dict() \
                      if 'population_weight' in nodes_df.columns else {}
        
        objective = pulp.lpSum(
            node_weights.get(n, 1.0) * amenity_weights.get(a, 1.0) * y[(n, a)]
            for n in demand_nodes
            for a in window['amenities'].keys()
        )
        prob += objective
        
        # Constraints: Coverage linking
        for n in demand_nodes:
            for a in window['amenities'].keys():
                threshold = distance_thresholds.get(a, 1000)
                
                nearby = [
                    i for i in local_candidates
                    if (i, a) in x and distance_cache.get((n, i), float('inf')) <= threshold
                ]
                
                if nearby:
                    prob += y[(n, a)] <= pulp.lpSum(x[(i, a)] for i in nearby)
                else:
                    prob += y[(n, a)] == 0
        
        # Constraints: Maintain amenity counts (swap-only)
        for amenity_type, nodes_to_relocate in window['amenities'].items():
            fixed_count = len(nodes_to_relocate)
            prob += (
                pulp.lpSum(x[(i, amenity_type)] for i in local_candidates if (i, amenity_type) in x)
                == fixed_count
            )
        
        var_mapping = {
            'x': x,
            'y': y,
            'demand_nodes': demand_nodes,
            'local_candidates': local_candidates
        }
        
        return prob, var_mapping
    
    def _extract_refined_candidate(
        self,
        original: Dict[str, List[int]],
        problem: pulp.LpProblem,
        var_mapping: Dict,
        window: Dict
    ) -> Dict[str, List[int]]:
        """Extract refined candidate from MILP solution."""
        refined = {k: list(v) for k, v in original.items()}  # Deep copy
        
        x = var_mapping['x']
        
        # Update placements for optimized amenities
        for amenity_type in window['amenities'].keys():
            new_placements = []
            
            # Keep placements outside window
            for node in original.get(amenity_type, []):
                if node not in window['amenities'].get(amenity_type, []):
                    new_placements.append(node)
            
            # Add MILP-selected placements
            for var_key, var in x.items():
                if var_key[1] == amenity_type and pulp.value(var) == 1:
                    new_placements.append(var_key[0])
            
            refined[amenity_type] = new_placements
        
        return refined
    
    def _estimate_fitness_improvement(
        self,
        original: Dict[str, List[int]],
        refined: Dict[str, List[int]],
        window: Dict,
        nodes_df: pd.DataFrame,
        distance_cache: Dict,
        amenity_weights: Dict[str, float]
    ) -> float:
        """Estimate fitness of refined candidate (fast approximation)."""
        # Focus on window hexagons for speed
        if window['hexagons'] and 'h3_08' in nodes_df.columns:
            eval_nodes = nodes_df[
                nodes_df['h3_08'].isin(window['hexagons'])
            ]['osmid'].values
        else:
            eval_nodes = nodes_df['osmid'].values[:500]  # Sample
        
        fitness = 0.0
        
        for node in eval_nodes:
            for amenity_type, placements in refined.items():
                weight = amenity_weights.get(amenity_type, 1.0)
                
                # Find nearest placement
                min_dist = float('inf')
                for placed in placements:
                    dist = distance_cache.get((node, placed), float('inf'))
                    min_dist = min(min_dist, dist)
                
                if min_dist < float('inf'):
                    fitness += weight / (min_dist + 1)
        
        return fitness
    
    def _count_relocations(
        self,
        original: Dict[str, List[int]],
        refined: Dict[str, List[int]]
    ) -> Dict[str, int]:
        """Count how many amenities were relocated per type."""
        relocated = {}
        
        for amenity_type in original.keys():
            orig_set = set(original.get(amenity_type, []))
            ref_set = set(refined.get(amenity_type, []))
            
            changes = len(orig_set.symmetric_difference(ref_set)) // 2
            if changes > 0:
                relocated[amenity_type] = changes
        
        return relocated
    
    def get_statistics(self) -> Dict:
        """Get refinement statistics."""
        stats = dict(self.stats)
        
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_solves'] / stats['total_attempts']
            stats['improvement_rate'] = stats['improvements'] / stats['total_attempts']
            stats['avg_solve_time'] = stats['total_solve_time'] / stats['total_attempts']
            stats['avg_improvement'] = (
                stats['total_improvement'] / stats['improvements']
                if stats['improvements'] > 0 else 0.0
            )
            
            if self.config.enable_caching:
                total_cache_queries = stats['cache_hits'] + stats['cache_misses']
                stats['cache_hit_rate'] = (
                    stats['cache_hits'] / total_cache_queries
                    if total_cache_queries > 0 else 0.0
                )
        
        return stats
    
    def save_statistics(self, output_path: Path) -> None:
        """Save refinement statistics to file."""
        stats = self.get_statistics()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"MILP refinement statistics saved to {output_path}")
    
    def log_summary(self) -> None:
        """Log summary statistics."""
        stats = self.get_statistics()
        
        logger.info("=" * 60)
        logger.info("MILP Refinement Summary")
        logger.info("=" * 60)
        logger.info(f"Total attempts: {stats['total_attempts']}")
        logger.info(f"Successful solves: {stats['successful_solves']} ({stats.get('success_rate', 0)*100:.1f}%)")
        logger.info(f"Improvements: {stats['improvements']} ({stats.get('improvement_rate', 0)*100:.1f}%)")
        logger.info(f"Total improvement: {stats['total_improvement']:.4f}")
        logger.info(f"Avg improvement: {stats.get('avg_improvement', 0):.4f}")
        logger.info(f"Avg solve time: {stats.get('avg_solve_time', 0):.3f}s")
        
        if self.config.enable_caching:
            logger.info(f"Cache hits: {stats['cache_hits']}")
            logger.info(f"Cache misses: {stats['cache_misses']}")
            logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0)*100:.1f}%")
        
        logger.info("=" * 60)


# Integration helper functions for GA

def create_hybrid_refiner(config_dict: Dict) -> Optional[HybridMILPRefiner]:
    """
    Factory function to create hybrid refiner from config.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        HybridMILPRefiner instance or None if disabled
    """
    hybrid_config = config_dict.get('hybrid_milp', {})
    
    if not hybrid_config.get('enabled', False):
        return None
    
    if pulp is None:
        logger.warning("PuLP not available, hybrid MILP disabled")
        return None
    
    refiner_config = MILPRefinementConfig(
        enabled=True,
        time_limit_seconds=hybrid_config.get('time_limit_seconds', 2.0),
        max_amenities_to_relocate=hybrid_config.get('max_amenities_to_relocate', 4),
        max_hexagons_to_optimize=hybrid_config.get('max_hexagons_to_optimize', 3),
        selection_strategy=hybrid_config.get('selection_strategy', 'worst_hexagons'),
        apply_to_elite_only=hybrid_config.get('apply_to_elite_only', True),
        apply_every_n_generations=hybrid_config.get('apply_every_n_generations', 1),
        min_improvement_threshold=hybrid_config.get('min_improvement_threshold', 0.01),
        max_candidates_per_generation=hybrid_config.get('max_candidates_per_generation', 5)
    )
    
    logger.info("Hybrid GA-MILP refiner initialized")
    logger.info(f"  Strategy: {refiner_config.selection_strategy}")
    logger.info(f"  Max relocations: {refiner_config.max_amenities_to_relocate}")
    logger.info(f"  Time limit: {refiner_config.time_limit_seconds}s")
    
    return HybridMILPRefiner(refiner_config)
