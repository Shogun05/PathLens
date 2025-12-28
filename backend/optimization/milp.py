import pulp
import numpy as np

class MILPConstraintFilter:
    def __init__(self, budget=50000000):
        self.budget = budget
        self.costs = {"school": 2000000, "hospital": 5000000, "park": 500000}
        self.min_spacing = {"school": 500, "hospital": 2000, "park": 300}

    def optimize_feasibility(self, candidates):
        if not candidates: return []
        
        prob = pulp.LpProblem("Feasibility", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("keep", range(len(candidates)), cat="Binary")
        
        # Objective: Keep max amenities
        prob += pulp.lpSum([x[i] for i in range(len(candidates))])
        
        # Constraint: Budget
        total_cost = pulp.lpSum([self.costs.get(c['type'], 100000) * x[i] for i, c in enumerate(candidates)])
        prob += (total_cost <= self.budget)

        # Constraint: Spacing
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, c2 = candidates[i], candidates[j]
                if c1['type'] == c2['type']:
                    dist = np.sqrt((c1['x']-c2['x'])**2 + (c1['y']-c2['y'])**2)
                    if dist < self.min_spacing.get(c1['type'], 100):
                        prob += (x[i] + x[j] <= 1)

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return [candidates[i] for i in range(len(candidates)) if pulp.value(x[i]) == 1]