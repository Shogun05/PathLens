"""
MILP-based Amenity Placement Optimizer for PathLens.
"""

import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import networkx as nx

# Add project root for CityDataManager
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

try:
    import pulp
except ImportError:
    raise ImportError("PuLP is required for MILP optimization. Install with: pip install pulp")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MILPConfig:
    """Configuration for MILP optimization."""
    amenity_budgets: Dict[str, int]
    amenity_weights: Dict[str, float]
    distance_thresholds: Dict[str, float]
    time_limit_seconds: int = 300
    enable_equity: bool = False
    equity_min_coverage: float = 0.3
    gap_tolerance: float = 0.01
    use_population_weights: bool = True
    test_mode: bool = False

@dataclass
class MILPSolution:
    """Container for MILP solution results."""
    placements: Dict[str, List[int]]
    objective_value: float
    solver_status: str
    solve_time_seconds: float
    gap: float
    coverage_stats: Dict[str, float]
    total_coverage: float
    population_covered: float
    constraint_counts: Dict[str, int]
    timestamp: str

class MILPAmenityPlacer:
    """MILP-based optimizer for amenity placement."""
    
    def __init__(self, cdm: CityDataManager, config: dict):
        self.cdm = cdm
        self.config = config
        self.milp_config: Optional[MILPConfig] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.candidates_df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.Graph] = None
        self.distance_matrix: Optional[Dict] = None

    def load_data(self) -> None:
        nodes_path = self.cdm.baseline_nodes
        candidates_path = self.cdm.high_travel_nodes(self.cdm.mode)
        graph_path = self.cdm.processed_graph

        logger.info(f"Loading data for {self.cdm.city}...")
        self.nodes_df = pd.read_parquet(nodes_path)
        if 'osmid' not in self.nodes_df.columns and self.nodes_df.index.name is not None:
            self.nodes_df = self.nodes_df.reset_index()
        
        self.candidates_df = pd.read_csv(candidates_path)
        candidate_ids = set(self.candidates_df['osmid'].values)
        self.nodes_df['is_candidate'] = self.nodes_df['osmid'].isin(candidate_ids)
        
        logger.info("Loading graph for distance computation...")
        self.graph = nx.read_graphml(graph_path, node_type=int)
        for u, v, data in self.graph.edges(data=True):
            if 'length' in data:
                data['length'] = float(data['length'])
        
        if 'population_weight' not in self.nodes_df.columns:
            self.nodes_df['population_weight'] = 1.0

    def setup_milp_config(self, time_limit: int = 300, enable_equity: bool = False, test_mode: bool = False) -> None:
        opt_cfg = self.config.get('optimization', {})
        amenity_weights = self.config.get('amenity_weights', {})
        distance_thresholds = self.config.get('equity_thresholds', {k: 1000 for k in amenity_weights.keys()})
        
        amenity_budgets = opt_cfg.get('amenity_budgets', {k: 5 for k in amenity_weights.keys()})
        
        self.milp_config = MILPConfig(
            amenity_budgets=amenity_budgets,
            amenity_weights=amenity_weights,
            distance_thresholds=distance_thresholds,
            time_limit_seconds=time_limit,
            enable_equity=enable_equity,
            test_mode=test_mode
        )

    def precompute_distances(self, amenity_types: List[str], max_candidates: Optional[int] = None) -> None:
        demand_nodes = self.nodes_df['osmid'].values
        candidate_nodes = self.candidates_df['osmid'].values[:max_candidates] if max_candidates else self.candidates_df['osmid'].values
        
        self.distance_matrix = {n: {} for n in demand_nodes}
        logger.info(f"Computing distances from {len(candidate_nodes)} candidates...")
        
        for i, candidate in enumerate(candidate_nodes):
            if candidate not in self.graph: continue
            try:
                lengths = nx.single_source_dijkstra_path_length(self.graph, candidate, weight='length')
                for demand_node in demand_nodes:
                    if demand_node in lengths:
                        self.distance_matrix[demand_node][candidate] = lengths[demand_node]
            except nx.NetworkXError: continue

    def build_milp_model(self, amenity_types: List[str]) -> pulp.LpProblem:
        prob = pulp.LpProblem("AmenityPlacement", pulp.LpMaximize)
        demand_nodes = self.nodes_df['osmid'].values
        candidate_nodes = [c for c in self.candidates_df['osmid'].values if any(c in self.distance_matrix[n] for n in demand_nodes)]
        
        if self.milp_config.test_mode:
            demand_nodes = demand_nodes[:100]
            candidate_nodes = candidate_nodes[:50]

        x = pulp.LpVariable.dicts("place", ((i, a) for i in candidate_nodes for a in amenity_types), cat=pulp.LpBinary)
        y = pulp.LpVariable.dicts("covered", ((n, a) for n in demand_nodes for a in amenity_types), cat=pulp.LpBinary)
        
        node_weights = self.nodes_df.set_index('osmid')['population_weight'].to_dict()
        prob += pulp.lpSum(node_weights.get(n, 1.0) * self.milp_config.amenity_weights.get(a, 1.0) * y[n, a] 
                           for n in demand_nodes for a in amenity_types)
        
        for n in demand_nodes:
            for a in amenity_types:
                threshold = self.milp_config.distance_thresholds.get(a, 1000)
                nearby = [i for i in candidate_nodes if self.distance_matrix[n].get(i, float('inf')) <= threshold]
                if nearby: prob += y[n, a] <= pulp.lpSum(x[i, a] for i in nearby)
                else: prob += y[n, a] == 0

        for a in amenity_types:
            prob += pulp.lpSum(x[i, a] for i in candidate_nodes) <= self.milp_config.amenity_budgets.get(a, 5)

        self._constraint_counts = {'variables': len(x) + len(y)}
        return prob

    def solve(self, problem: pulp.LpProblem) -> MILPSolution:
        solver = pulp.PULP_CBC_CMD(timeLimit=self.milp_config.time_limit_seconds, gapRel=self.milp_config.gap_tolerance, msg=True)
        start_time = datetime.now()
        status = problem.solve(solver)
        solve_time = (datetime.now() - start_time).total_seconds()
        
        placements = self._extract_placements(problem)
        coverage_stats = self._compute_coverage_stats(problem, placements)
        
        return MILPSolution(
            placements=placements,
            objective_value=pulp.value(problem.objective),
            solver_status=pulp.LpStatus[status],
            solve_time_seconds=solve_time,
            gap=solver.actualGap if hasattr(solver, 'actualGap') else 0.0,
            coverage_stats=coverage_stats['per_amenity'],
            total_coverage=coverage_stats['total'],
            population_covered=coverage_stats['population_weighted'],
            constraint_counts=self._constraint_counts,
            timestamp=datetime.now().isoformat()
        )

    def _extract_placements(self, problem: pulp.LpProblem) -> Dict[str, List[int]]:
        placements = {}
        for var in problem.variables():
            if var.name.startswith("place_") and pulp.value(var) == 1:
                parts = var.name.replace("place_(", "").replace(")", "").split(",_")
                node_id = int(parts[0])
                amenity_type = parts[1].strip("'\"")
                placements.setdefault(amenity_type, []).append(node_id)
        return placements

    def _compute_coverage_stats(self, problem: pulp.LpProblem, placements: Dict[str, List[int]]) -> Dict:
        demand_nodes = self.nodes_df['osmid'].values
        total_pop = self.nodes_df['population_weight'].sum()
        stats = {'per_amenity': {}, 'total': 0.0, 'population_weighted': 0.0}
        
        covered_pop = 0.0
        for n in demand_nodes:
            covered = False
            for var in problem.variables():
                if var.name.startswith(f"covered_({n},") and pulp.value(var) == 1:
                    covered = True
                    break
            if covered:
                covered_pop += self.nodes_df[self.nodes_df['osmid'] == n]['population_weight'].iloc[0]
        
        stats['population_weighted'] = (covered_pop / total_pop) * 100 if total_pop > 0 else 0
        return stats

    def export_solution(self, solution: MILPSolution) -> None:
        out_path = self.cdm.optimized_dir(self.cdm.mode) / "milp_results"
        out_path.mkdir(parents=True, exist_ok=True)
        
        with open(out_path / "milp_solution.json", 'w') as f:
            json.dump(asdict(solution), f, indent=2)
        logger.info(f"Solution saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="MILP-based amenity placement optimization")
    parser.add_argument('--city', default='bangalore', help='City to process')
    parser.add_argument('--mode', default='ga_milp', help='Optimization mode')
    parser.add_argument('--time-limit', type=int, default=300)
    args = parser.parse_args()
    
    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    cfg = cdm.load_config()
    
    optimizer = MILPAmenityPlacer(cdm, cfg)
    optimizer.load_data()
    
    amenity_types = list(cfg.get('amenity_weights', {}).keys())
    optimizer.setup_milp_config(time_limit=args.time_limit)
    optimizer.precompute_distances(amenity_types)
    
    problem = optimizer.build_milp_model(amenity_types)
    solution = optimizer.solve(problem)
    optimizer.export_solution(solution)

if __name__ == "__main__":
    main()
