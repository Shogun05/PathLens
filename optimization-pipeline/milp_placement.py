"""
MILP-based Amenity Placement Optimizer

Implements a Mixed Integer Linear Programming approach to facility location
for optimal amenity placement in urban pedestrian networks.

Mathematical Formulation:
    Sets:
        N: demand nodes (network nodes with population weights)
        C: candidate placement nodes (high travel time nodes)
        A: amenity types
    
    Decision Variables:
        x[i,a] ∈ {0,1}: place amenity type a at candidate node i
        y[n,a] ∈ {0,1}: demand node n is covered by amenity type a
    
    Objective:
        maximize Σ_n Σ_a (P[n] * w[a] * y[n,a])
        where P[n] = population weight, w[a] = amenity importance
    
    Constraints:
        1. Coverage: y[n,a] ≤ Σ_{i ∈ C | d[n,i] ≤ D[a]} x[i,a]
        2. Budget: Σ_i x[i,a] ≤ B[a]
        3. Binary domains
        4. Optional: Equity constraints per H3 hexagon
"""

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

try:
    import pulp
except ImportError:
    raise ImportError(
        "PuLP is required for MILP optimization. Install with: pip install pulp"
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MILPConfig:
    """Configuration for MILP optimization."""
    amenity_budgets: Dict[str, int]  # Max new amenities per type
    amenity_weights: Dict[str, float]  # Importance weights
    distance_thresholds: Dict[str, float]  # Coverage thresholds (meters)
    time_limit_seconds: int = 300
    enable_equity: bool = False
    equity_min_coverage: float = 0.3  # Minimum coverage per hexagon (if enabled)
    gap_tolerance: float = 0.01  # MIP gap tolerance
    use_population_weights: bool = True
    test_mode: bool = False  # Limit problem size for testing


@dataclass
class MILPSolution:
    """Container for MILP solution results."""
    placements: Dict[str, List[int]]  # amenity_type -> [node_ids]
    objective_value: float
    solver_status: str
    solve_time_seconds: float
    gap: float
    coverage_stats: Dict[str, float]  # Per-amenity coverage percentages
    total_coverage: float  # Overall weighted coverage
    population_covered: float  # Population-weighted coverage
    constraint_counts: Dict[str, int]
    timestamp: str


class MILPAmenityPlacer:
    """
    MILP-based optimizer for amenity placement using facility location formulation.
    
    This optimizer provides exact solutions (within tolerance) to the amenity
    placement problem, complementing the genetic algorithm approach.
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize MILP optimizer.
        
        Args:
            config_path: Path to PathLens configuration file
        """
        with open(config_path, 'r') as f:
            self.global_config = yaml.safe_load(f)
        
        self.milp_config: Optional[MILPConfig] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.candidates_df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.Graph] = None
        self.distance_matrix: Optional[Dict] = None
        
    def load_data(
        self,
        nodes_path: str,
        candidates_path: str,
        graph_path: Optional[str] = None
    ) -> None:
        """
        Load required data artifacts.
        
        Args:
            nodes_path: Path to nodes_with_scores.parquet
            candidates_path: Path to high_travel_time_nodes.csv
            graph_path: Optional path to graph.graphml for distance computation
        """
        logger.info("Loading node data...")
        self.nodes_df = pd.read_parquet(nodes_path)
        
        # Reset index if osmid is in index
        if 'osmid' not in self.nodes_df.columns and self.nodes_df.index.name is not None:
            self.nodes_df = self.nodes_df.reset_index()
        
        logger.info(f"Loaded {len(self.nodes_df)} demand nodes")
        
        logger.info("Loading candidate nodes...")
        self.candidates_df = pd.read_csv(candidates_path)
        logger.info(f"Loaded {len(self.candidates_df)} candidate placement nodes")
        
        # Ensure candidate nodes are in the nodes dataframe
        candidate_ids = set(self.candidates_df['osmid'].values)
        self.nodes_df['is_candidate'] = self.nodes_df['osmid'].isin(candidate_ids)
        
        if graph_path:
            logger.info("Loading graph for distance computation...")
            self.graph = nx.read_graphml(graph_path, node_type=int)
            
            # Convert edge weights to floats
            for u, v, data in self.graph.edges(data=True):
                if 'length' in data:
                    data['length'] = float(data['length'])
            
            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes, "
                       f"{self.graph.number_of_edges()} edges")
        
        # Initialize population weights
        if 'population_weight' not in self.nodes_df.columns:
            logger.info("Population weights not found, using uniform weights")
            self.nodes_df['population_weight'] = 1.0
        else:
            logger.info("Using existing population weights")
    
    def setup_milp_config(
        self,
        amenity_budgets: Optional[Dict[str, int]] = None,
        time_limit: int = 300,
        enable_equity: bool = False,
        test_mode: bool = False
    ) -> None:
        """
        Configure MILP parameters.
        
        Args:
            amenity_budgets: Custom budgets per amenity type (default from config)
            time_limit: Solver time limit in seconds
            enable_equity: Enable H3 hexagon equity constraints
            test_mode: Reduce problem size for testing
        """
        # Extract weights and thresholds from global config
        amenity_weights = self.global_config.get('amenity_weights', {})
        equity_thresholds = self.global_config.get('equity_thresholds', {})
        
        # Default budgets if not provided
        if amenity_budgets is None:
            amenity_budgets = self.global_config.get('milp', {}).get(
                'amenity_budgets',
                {k: 5 for k in amenity_weights.keys()}  # Default 5 per type
            )
        
        self.milp_config = MILPConfig(
            amenity_budgets=amenity_budgets,
            amenity_weights=amenity_weights,
            distance_thresholds=equity_thresholds,
            time_limit_seconds=time_limit,
            enable_equity=enable_equity,
            test_mode=test_mode
        )
        
        logger.info(f"MILP Configuration:")
        logger.info(f"  Amenity budgets: {amenity_budgets}")
        logger.info(f"  Time limit: {time_limit}s")
        logger.info(f"  Equity constraints: {enable_equity}")
        logger.info(f"  Test mode: {test_mode}")
    
    def precompute_distances(
        self,
        amenity_types: List[str],
        max_candidates: Optional[int] = None
    ) -> None:
        """
        Precompute network distances from demand nodes to candidate nodes.
        
        Args:
            amenity_types: List of amenity types to optimize
            max_candidates: Limit candidates for testing (None = all)
        """
        if self.graph is None:
            raise ValueError("Graph must be loaded for distance computation")
        
        logger.info("Precomputing distance matrix...")
        
        # Get node sets
        demand_nodes = self.nodes_df['osmid'].values
        candidate_nodes = self.candidates_df['osmid'].values
        
        if max_candidates:
            candidate_nodes = candidate_nodes[:max_candidates]
            logger.info(f"Limited to {max_candidates} candidates (test mode)")
        
        # Initialize distance dictionary: d[demand_id][candidate_id] = distance
        self.distance_matrix = {n: {} for n in demand_nodes}
        
        # Compute shortest paths from each candidate to all demand nodes
        total_pairs = len(candidate_nodes) * len(demand_nodes)
        logger.info(f"Computing {total_pairs:,} distances...")
        
        for i, candidate in enumerate(candidate_nodes):
            if i % 50 == 0:
                logger.info(f"  Processing candidate {i+1}/{len(candidate_nodes)}...")
            
            if candidate not in self.graph:
                continue
            
            # Single-source shortest path from candidate
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    self.graph,
                    candidate,
                    weight='length'
                )
                
                # Store distances to demand nodes
                for demand_node in demand_nodes:
                    if demand_node in lengths:
                        self.distance_matrix[demand_node][candidate] = lengths[demand_node]
            
            except nx.NetworkXError:
                logger.warning(f"Candidate {candidate} unreachable, skipping")
                continue
        
        logger.info("Distance matrix precomputed")
    
    def build_milp_model(self, amenity_types: List[str]) -> pulp.LpProblem:
        """
        Construct the MILP facility location model.
        
        Args:
            amenity_types: List of amenity types to place
            
        Returns:
            PuLP optimization problem
        """
        if self.milp_config is None:
            raise ValueError("MILP config not set. Call setup_milp_config() first")
        
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not computed. Call precompute_distances() first")
        
        logger.info("Building MILP model...")
        
        # Filter amenity types to those in config
        amenity_types = [
            a for a in amenity_types 
            if a in self.milp_config.amenity_weights
        ]
        
        if not amenity_types:
            raise ValueError("No valid amenity types specified")
        
        # Create optimization problem
        prob = pulp.LpProblem("AmenityPlacement", pulp.LpMaximize)
        
        # Get node sets
        demand_nodes = self.nodes_df['osmid'].values
        candidate_nodes = [
            c for c in self.candidates_df['osmid'].values
            if any(c in self.distance_matrix[n] for n in demand_nodes)
        ]
        
        if self.milp_config.test_mode:
            demand_nodes = demand_nodes[:min(100, len(demand_nodes))]
            candidate_nodes = candidate_nodes[:min(50, len(candidate_nodes))]
            logger.info(f"Test mode: {len(demand_nodes)} demand, "
                       f"{len(candidate_nodes)} candidates")
        
        logger.info(f"Problem size: {len(demand_nodes)} demand nodes, "
                   f"{len(candidate_nodes)} candidates, {len(amenity_types)} amenity types")
        
        # Decision variables
        # x[i,a]: place amenity a at candidate i
        x = pulp.LpVariable.dicts(
            "place",
            ((i, a) for i in candidate_nodes for a in amenity_types),
            cat=pulp.LpBinary
        )
        
        # y[n,a]: demand node n is covered by amenity a
        y = pulp.LpVariable.dicts(
            "covered",
            ((n, a) for n in demand_nodes for a in amenity_types),
            cat=pulp.LpBinary
        )
        
        # Objective: maximize population-weighted coverage
        node_weights = self.nodes_df.set_index('osmid')['population_weight'].to_dict()
        amenity_weights = self.milp_config.amenity_weights
        
        objective = pulp.lpSum(
            node_weights.get(n, 1.0) * amenity_weights.get(a, 1.0) * y[n, a]
            for n in demand_nodes
            for a in amenity_types
        )
        prob += objective, "MaximizeWeightedCoverage"
        
        # Constraint 1: Coverage linking
        logger.info("Adding coverage constraints...")
        coverage_count = 0
        for n in demand_nodes:
            for a in amenity_types:
                threshold = self.milp_config.distance_thresholds.get(a, 1000)
                
                # Find candidates within threshold
                nearby_candidates = [
                    i for i in candidate_nodes
                    if self.distance_matrix[n].get(i, float('inf')) <= threshold
                ]
                
                if nearby_candidates:
                    prob += (
                        y[n, a] <= pulp.lpSum(x[i, a] for i in nearby_candidates),
                        f"Coverage_{n}_{a}"
                    )
                    coverage_count += 1
                else:
                    # Node cannot be covered by this amenity type
                    prob += y[n, a] == 0, f"NoCoverage_{n}_{a}"
        
        logger.info(f"Added {coverage_count} coverage constraints")
        
        # Constraint 2: Budget limits
        logger.info("Adding budget constraints...")
        for a in amenity_types:
            budget = self.milp_config.amenity_budgets.get(a, 5)
            prob += (
                pulp.lpSum(x[i, a] for i in candidate_nodes) <= budget,
                f"Budget_{a}"
            )
        
        # Optional Constraint 3: Equity constraints
        if self.milp_config.enable_equity and 'h3_08' in self.nodes_df.columns:
            logger.info("Adding equity constraints...")
            hexagons = self.nodes_df['h3_08'].unique()
            min_coverage = self.milp_config.equity_min_coverage
            
            equity_count = 0
            for hex_id in hexagons:
                hex_nodes = self.nodes_df[
                    self.nodes_df['h3_08'] == hex_id
                ]['osmid'].values
                
                hex_nodes = [n for n in hex_nodes if n in demand_nodes]
                
                if hex_nodes:
                    # Minimum weighted coverage per hexagon
                    hex_coverage = pulp.lpSum(
                        amenity_weights.get(a, 1.0) * y[n, a]
                        for n in hex_nodes
                        for a in amenity_types
                    )
                    
                    prob += (
                        hex_coverage >= min_coverage * len(hex_nodes) * len(amenity_types),
                        f"Equity_{hex_id}"
                    )
                    equity_count += 1
            
            logger.info(f"Added {equity_count} equity constraints")
        
        # Store for analysis
        self._constraint_counts = {
            'coverage': coverage_count,
            'budget': len(amenity_types),
            'equity': equity_count if self.milp_config.enable_equity else 0,
            'variables': len(x) + len(y)
        }
        
        logger.info(f"Model built: {len(x) + len(y)} variables, "
                   f"{sum(self._constraint_counts.values())} constraints")
        
        return prob
    
    def solve(self, problem: pulp.LpProblem) -> MILPSolution:
        """
        Solve the MILP problem.
        
        Args:
            problem: PuLP optimization problem
            
        Returns:
            MILPSolution with results
        """
        logger.info("Solving MILP...")
        
        # Configure solver
        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.milp_config.time_limit_seconds,
            gapRel=self.milp_config.gap_tolerance,
            msg=True
        )
        
        # Solve
        start_time = datetime.now()
        status = problem.solve(solver)
        solve_time = (datetime.now() - start_time).total_seconds()
        
        status_str = pulp.LpStatus[status]
        logger.info(f"Solver status: {status_str}")
        logger.info(f"Solve time: {solve_time:.2f}s")
        
        if status != pulp.LpStatusOptimal and status != pulp.LpStatusNotSolved:
            logger.warning(f"Solution may not be optimal: {status_str}")
        
        # Extract solution
        placements = self._extract_placements(problem)
        coverage_stats = self._compute_coverage_stats(problem, placements)
        
        solution = MILPSolution(
            placements=placements,
            objective_value=pulp.value(problem.objective),
            solver_status=status_str,
            solve_time_seconds=solve_time,
            gap=solver.actualGap if hasattr(solver, 'actualGap') else 0.0,
            coverage_stats=coverage_stats['per_amenity'],
            total_coverage=coverage_stats['total'],
            population_covered=coverage_stats['population_weighted'],
            constraint_counts=self._constraint_counts,
            timestamp=datetime.now().isoformat()
        )
        
        return solution
    
    def _extract_placements(self, problem: pulp.LpProblem) -> Dict[str, List[int]]:
        """Extract amenity placements from solved problem."""
        placements = {}
        
        for var in problem.variables():
            if var.name.startswith("place_") and pulp.value(var) == 1:
                # Variable name format: place_(node_id, amenity_type)
                parts = var.name.replace("place_(", "").replace(")", "").split(",_")
                node_id = int(parts[0])
                amenity_type = parts[1].strip("'\"")
                
                if amenity_type not in placements:
                    placements[amenity_type] = []
                placements[amenity_type].append(node_id)
        
        for amenity_type, nodes in placements.items():
            logger.info(f"  {amenity_type}: {len(nodes)} placements")
        
        return placements
    
    def _compute_coverage_stats(
        self,
        problem: pulp.LpProblem,
        placements: Dict[str, List[int]]
    ) -> Dict:
        """Compute coverage statistics from solution."""
        stats = {'per_amenity': {}, 'total': 0.0, 'population_weighted': 0.0}
        
        demand_nodes = self.nodes_df['osmid'].values
        total_demand = len(demand_nodes)
        total_population = self.nodes_df['population_weight'].sum()
        
        covered_count = 0
        covered_population = 0.0
        
        for amenity_type in placements.keys():
            amenity_covered = 0
            
            for var in problem.variables():
                if var.name.startswith(f"covered_") and amenity_type in var.name:
                    if pulp.value(var) == 1:
                        amenity_covered += 1
            
            coverage_pct = (amenity_covered / total_demand) * 100
            stats['per_amenity'][amenity_type] = coverage_pct
        
        # Count unique nodes covered by any amenity
        for n in demand_nodes:
            node_covered = False
            for var in problem.variables():
                if var.name.startswith(f"covered_({n},") and pulp.value(var) == 1:
                    node_covered = True
                    node_weight = self.nodes_df[
                        self.nodes_df['osmid'] == n
                    ]['population_weight'].iloc[0]
                    covered_population += node_weight
                    break
            
            if node_covered:
                covered_count += 1
        
        stats['total'] = (covered_count / total_demand) * 100
        stats['population_weighted'] = (covered_population / total_population) * 100
        
        return stats
    
    def export_solution(
        self,
        solution: MILPSolution,
        output_dir: str,
        create_geojson: bool = True
    ) -> None:
        """
        Export solution to files.
        
        Args:
            solution: MILP solution object
            output_dir: Output directory path
            create_geojson: Create GeoJSON visualization
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export JSON solution
        json_path = output_path / "milp_solution.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(solution), f, indent=2)
        logger.info(f"Solution saved to {json_path}")
        
        # Export summary metrics
        summary = {
            'objective_value': solution.objective_value,
            'solver_status': solution.solver_status,
            'solve_time_seconds': solution.solve_time_seconds,
            'gap': solution.gap,
            'total_coverage_pct': solution.total_coverage,
            'population_coverage_pct': solution.population_covered,
            'placements_by_type': {
                k: len(v) for k, v in solution.placements.items()
            },
            'coverage_by_type': solution.coverage_stats,
            'problem_size': solution.constraint_counts
        }
        
        summary_path = output_path / "milp_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        
        # Create GeoJSON if requested
        if create_geojson and 'x' in self.nodes_df.columns:
            self._create_solution_geojson(solution, output_path)
    
    def _create_solution_geojson(
        self,
        solution: MILPSolution,
        output_path: Path
    ) -> None:
        """Create GeoJSON visualization of solution."""
        features = []
        
        for amenity_type, node_ids in solution.placements.items():
            for node_id in node_ids:
                node_data = self.nodes_df[self.nodes_df['osmid'] == node_id]
                
                if len(node_data) == 0:
                    continue
                
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [
                            float(node_data.iloc[0]['x']),
                            float(node_data.iloc[0]['y'])
                        ]
                    },
                    'properties': {
                        'osmid': int(node_id),
                        'amenity_type': amenity_type,
                        'placement_method': 'milp',
                        'original_travel_time': float(
                            node_data.iloc[0].get('min_travel_time_minutes', 0)
                        )
                    }
                }
                features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        geojson_path = output_path / "milp_placements.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        logger.info(f"GeoJSON saved to {geojson_path}")


def main():
    """CLI entry point for MILP optimization."""
    parser = argparse.ArgumentParser(
        description="MILP-based amenity placement optimization"
    )
    
    parser.add_argument(
        '--nodes-scores',
        default='../data/analysis/nodes_with_scores.parquet',
        help='Path to nodes_with_scores.parquet'
    )
    parser.add_argument(
        '--high-travel',
        default='../data/optimization/high_travel_time_nodes.csv',
        help='Path to candidate nodes CSV'
    )
    parser.add_argument(
        '--graph',
        default='../data/processed/graph.graphml',
        help='Path to graph for distance computation'
    )
    parser.add_argument(
        '--output-dir',
        default='../data/optimization/milp_results',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        default='../config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--amenities',
        nargs='+',
        help='Amenity types to optimize (default: all from config)'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=300,
        help='Solver time limit in seconds'
    )
    parser.add_argument(
        '--enable-equity',
        action='store_true',
        help='Enable H3 hexagon equity constraints'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with reduced problem size'
    )
    parser.add_argument(
        '--max-candidates',
        type=int,
        help='Limit number of candidate nodes (for testing)'
    )
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = MILPAmenityPlacer(args.config)
    
    # Load data
    optimizer.load_data(
        nodes_path=args.nodes_scores,
        candidates_path=args.high_travel,
        graph_path=args.graph
    )
    
    # Determine amenity types
    if args.amenities:
        amenity_types = args.amenities
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        amenity_types = list(config.get('amenity_weights', {}).keys())
    
    logger.info(f"Optimizing for amenity types: {amenity_types}")
    
    # Setup MILP configuration
    optimizer.setup_milp_config(
        time_limit=args.time_limit,
        enable_equity=args.enable_equity,
        test_mode=args.test_mode
    )
    
    # Precompute distances
    optimizer.precompute_distances(
        amenity_types=amenity_types,
        max_candidates=args.max_candidates
    )
    
    # Build and solve model
    problem = optimizer.build_milp_model(amenity_types)
    solution = optimizer.solve(problem)
    
    # Export results
    optimizer.export_solution(solution, args.output_dir)
    
    logger.info("=" * 60)
    logger.info("MILP Optimization Complete")
    logger.info(f"Objective Value: {solution.objective_value:.2f}")
    logger.info(f"Total Coverage: {solution.total_coverage:.1f}%")
    logger.info(f"Population Coverage: {solution.population_covered:.1f}%")
    logger.info(f"Solve Time: {solution.solve_time_seconds:.2f}s")
    logger.info(f"Status: {solution.solver_status}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
