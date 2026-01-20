"""
Example: Integrating Hybrid MILP Refiner into Genetic Algorithm

This example shows how to integrate the HybridMILPRefiner into an existing
GA evolution loop. Add this code to your hybrid_ga.py main loop.
"""

import yaml
from pathlib import Path
from optimization.hybrid_milp_refiner import create_hybrid_refiner, MILPRefinementConfig

# Example integration in hybrid_ga.py:

def run_ga_with_milp_refinement(
    population,
    nodes_df,
    distance_cache,
    amenity_weights,
    distance_thresholds,
    candidate_pool,
    num_generations,
    config
):
    """
    Example GA loop with integrated MILP refinement.
    
    This should be adapted to your existing hybrid_ga.py structure.
    """
    
    # Initialize hybrid refiner from config
    hybrid_refiner = create_hybrid_refiner(config)
    
    if hybrid_refiner:
        print("Hybrid GA-MILP mode enabled")
    else:
        print("Pure GA mode (MILP disabled)")
    
    best_fitness_history = []
    
    for generation in range(num_generations):
        print(f"\n=== Generation {generation} ===")
        
        # 1. Evaluate population
        fitness_scores = []
        for candidate in population:
            fitness = evaluate_fitness(candidate, nodes_df, distance_cache, amenity_weights)
            fitness_scores.append(fitness)
        
        # 2. Sort by fitness
        sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        best_fitness = fitness_scores[0]
        best_fitness_history.append(best_fitness)
        print(f"Best fitness: {best_fitness:.4f}")
        
        # 3. Apply MILP refinement to elite candidates (if enabled)
        if hybrid_refiner:
            milp_improved_count = 0
            
            for rank, candidate_idx in enumerate(sorted_indices):
                # Check if this candidate should be refined
                if not hybrid_refiner.should_refine_candidate(
                    generation, rank, len(population)
                ):
                    continue
                
                candidate = population[candidate_idx]
                current_fitness = fitness_scores[candidate_idx]
                
                # Apply MILP refinement
                refined_candidate, result = hybrid_refiner.refine_candidate(
                    candidate=candidate,
                    nodes_df=nodes_df,
                    distance_cache=distance_cache,
                    amenity_weights=amenity_weights,
                    distance_thresholds=distance_thresholds,
                    candidate_pool=candidate_pool,
                    current_fitness=current_fitness
                )
                
                # Update population if improved
                if result.improved:
                    population[candidate_idx] = refined_candidate
                    fitness_scores[candidate_idx] = result.refined_fitness
                    milp_improved_count += 1
                    print(f"  MILP improved candidate {rank}: "
                          f"{current_fitness:.4f} -> {result.refined_fitness:.4f}")
            
            if milp_improved_count > 0:
                print(f"MILP refined {milp_improved_count} candidates this generation")
        
        # 4. Selection, crossover, mutation (existing GA operators)
        # ... your existing GA code ...
        
        # 5. Checkpoint (save generation state for resumability)
        if generation % 5 == 0:
            checkpoint_path = Path(f"optimization/runs/checkpoint_gen_{generation}.json")
            save_checkpoint(population, fitness_scores, generation, checkpoint_path)
    
    # Final statistics
    if hybrid_refiner:
        hybrid_refiner.log_summary()
        stats_path = Path("optimization/runs/milp_refinement_stats.json")
        hybrid_refiner.save_statistics(stats_path)
    
    return population[0], best_fitness_history


def load_checkpoint(checkpoint_path: Path):
    """Load GA state from checkpoint for resumability."""
    import json
    
    if not checkpoint_path.exists():
        return None
    
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    print(f"Resuming from generation {checkpoint['generation']}")
    return checkpoint


def save_checkpoint(population, fitness_scores, generation, checkpoint_path: Path):
    """Save GA state for resumability."""
    import json
    
    checkpoint = {
        'generation': generation,
        'population': population,
        'fitness_scores': fitness_scores,
        'timestamp': str(Path.ctime(checkpoint_path))
    }
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_path}")


# Example config.yaml structure:
"""
hybrid_milp:
  enabled: true
  time_limit_seconds: 2.0
  max_amenities_to_relocate: 4
  max_hexagons_to_optimize: 3
  selection_strategy: "worst_hexagons"
  apply_to_elite_only: true
  apply_every_n_generations: 1
  max_candidates_per_generation: 5
  min_improvement_threshold: 0.01
  enable_caching: true
  cache_dir: "../data/cache/milp"
"""
