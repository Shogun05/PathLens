# Personalized Neuro Mixed Logit Regression (PNMLR) for Amenity Placement Optimization

## Abstract

We present an adaptation of Personalized Neuro Mixed Logit Regression (PNMLR) for urban amenity placement optimization. Unlike conventional approaches that optimize for aggregate accessibility metrics, our method models heterogeneous user preferences through synthetic population profiles generated via Dirichlet sampling. A neural utility model learns to predict accessibility value conditioned on these preference profiles, enabling optimization decisions that improve walkability for diverse population segments rather than a single representative agent.

---

## 1. Introduction

Traditional amenity placement optimization minimizes aggregate distance metrics or maximizes coverage for an "average" user. This approach fails to capture preference heterogeneity—different population segments value different amenity types differently. We address this limitation by:

1. Generating synthetic user profiles representing diverse preference patterns
2. Training a neural network to predict accessibility utility conditioned on user preferences
3. Evaluating candidate placements by averaging predicted utilities across all profiles

---

## 2. Methodology

### 2.1 Synthetic User Profile Generation

We generate $P = 20$ synthetic user profiles using the Dirichlet distribution, which produces valid probability vectors over $A$ amenity types. Let $\boldsymbol{\alpha} \in \mathbb{R}^A$ be the concentration parameter vector.

$$\mathbf{w}_p \sim \text{Dir}(\boldsymbol{\alpha}), \quad p \in \{1, \ldots, P\}$$

where $\mathbf{w}_p = (w_{p,1}, \ldots, w_{p,A})$ satisfies $\sum_{a=1}^{A} w_{p,a} = 1$ and $w_{p,a} \geq 0$.

To capture diverse preference patterns, we generate profiles from three archetypes:

| Archetype | Count | Concentration $\alpha$ | Characteristic |
|-----------|-------|------------------------|----------------|
| Balanced | $P/3$ | $\alpha_a = 2.0$ | Uniform preferences |
| Focused | $P/3$ | $\alpha_{\text{peak}} = 3.0$, others $= 0.5$ | Strong single preference |
| Moderate | $P/3$ | $\alpha_a = 1.0$ | Moderate variation |

### 2.2 Utility Function Formulation

For a node $n$ with distance vector $\mathbf{d}_n = (d_{n,1}, \ldots, d_{n,A})$ to the nearest amenities of each type, and user profile $\mathbf{w}_p$, we compute utility using exponential distance decay:

$$U(n, \mathbf{w}_p) = \sum_{a=1}^{A} w_{p,a} \cdot \exp\left(-\frac{d_{n,a}}{\lambda}\right)$$

where $\lambda = 2000$ meters is the decay constant governing diminishing accessibility returns.

### 2.3 Neural Utility Model Architecture

We train a Multi-Layer Perceptron (MLP) to approximate the utility function:

$$\hat{U}(n, \mathbf{w}_p) = f_\theta(\mathbf{x}_n \oplus \mathbf{w}_p)$$

where:
- $\mathbf{x}_n \in \mathbb{R}^F$ is the feature vector for node $n$
- $\mathbf{w}_p \in \mathbb{R}^A$ is the profile weight vector
- $\oplus$ denotes concatenation
- $f_\theta$ is the neural network with parameters $\theta$

**Architecture:**
$$\mathbb{R}^{F+A} \xrightarrow{\text{Linear}} \mathbb{R}^{64} \xrightarrow{\text{ReLU}} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} \mathbb{R}^{1}$$

**Feature Vector $\mathbf{x}_n$ ($F = 12$ dimensions):**

| Feature | Description |
|---------|-------------|
| $d_{n,a}^{\text{norm}}$ | Normalized distance to amenity $a$ (7 features) |
| $(x_n, y_n)$ | Normalized coordinates |
| $\text{deg}(n)$ | Node degree in road network |
| $t_n$ | Travel time metric |
| $s_n$ | Existing accessibility score |

**Training Objective:**
$$\mathcal{L}(\theta) = \frac{1}{NP} \sum_{n=1}^{N} \sum_{p=1}^{P} \left( \hat{U}(n, \mathbf{w}_p) - U(n, \mathbf{w}_p) \right)^2$$

### 2.4 Integration with Genetic Algorithm

The PNMLR model integrates with the PathLens hybrid genetic algorithm through two hooks:

**Precompute Hook** (called once at initialization):
```
Algorithm: PNMLR_PRECOMPUTE(nodes, model, profiles)
Input: N nodes, trained model f_θ, P profiles
Output: Utility map U: node_id → average utility

1. Extract feature matrix X ∈ ℝ^(N×F)
2. Normalize features: X_norm = (X - X_min) / (X_max - X_min)
3. For each profile p ∈ {1,...,P}:
     U_p = f_θ(X_norm ⊕ W_p)
4. Compute average: Ū = (1/P) Σ_p U_p
5. Return {n_i → Ū_i}
```

**Evaluate Hook** (called per candidate):
```
Algorithm: PNMLR_EVALUATE(candidate, utility_map, context)
Input: Candidate placement C, utility map U, GA context
Output: Fitness score

1. total_utility ← 0
2. For each (amenity_type, placed_nodes) in C:
     weight ← amenity_weights[amenity_type]
     For each node n in placed_nodes:
         total_utility ← total_utility + U[n] × weight

3. diversity_penalty ← COMPUTE_DIVERSITY_PENALTY(C)
4. proximity_penalty ← COMPUTE_PROXIMITY_PENALTY(C)
5. travel_penalty ← COMPUTE_TRAVEL_PENALTY(C)

6. fitness ← total_utility - diversity_penalty - proximity_penalty - 0.0005 × travel_penalty
7. Return max(fitness, 0)
```

### 2.5 Penalty Functions

**Diversity Penalty:** Penalizes same-type amenities placed within minimum spacing $\delta = 800$ meters.

$$P_{\text{div}} = \sum_{a=1}^{A} \sum_{i < j} \mathbf{1}[\|n_i - n_j\| < \delta] \cdot \left(1 - \frac{\|n_i - n_j\|}{\delta}\right)^2$$

**Proximity Penalty:** Penalizes placements within $\epsilon = 600$ meters of existing amenities.

$$P_{\text{prox}} = \sum_{n \in C} \mathbf{1}[d_n < \epsilon] \cdot 0.5 \cdot \left(1 - \frac{d_n}{\epsilon}\right)^2$$

---

## 3. Experimental Setup

### 3.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 77,018 |
| Profiles | 20 |
| Training samples | 1,540,360 |
| Epochs | 100 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Hidden dimensions | [64, 32] |
| Activation | ReLU |
| Decay constant $\lambda$ | 2000 m |

### 3.2 Amenity Types

| Amenity | Weight |
|---------|--------|
| School | 1.5 |
| Hospital | 2.5 |
| Pharmacy | 2.0 |
| Supermarket | 2.5 |
| Bus Station | 1.5 |
| Park | 1.2 |
| Bank | 1.8 |

---

## 4. Results

### 4.1 Training Performance

| Metric | Value |
|--------|-------|
| Final MSE Loss | 0.024 |
| Utility Range | [0.553, 0.702] |
| Mean Utility | 0.633 |
| Std Utility | 0.031 |

### 4.2 Profile Statistics

| Amenity | Mean Weight | Std | Range |
|---------|-------------|-----|-------|
| School | 0.175 | 0.165 | [0.004, 0.713] |
| Hospital | 0.150 | 0.116 | [0.000, 0.499] |
| Pharmacy | 0.146 | 0.149 | [0.002, 0.567] |
| Supermarket | 0.108 | 0.092 | [0.003, 0.339] |
| Bus Station | 0.173 | 0.142 | [0.013, 0.667] |
| Park | 0.142 | 0.197 | [0.000, 0.919] |
| Bank | 0.106 | 0.092 | [0.006, 0.284] |

---

## 5. Discussion

The PNMLR approach offers several advantages over traditional distance-based optimization:

1. **Heterogeneity Modeling:** Captures diverse accessibility preferences through profile-conditioned utility prediction

2. **Population Equity:** Averaging across profiles ensures placements benefit multiple user segments, not just the majority

3. **Scalable Inference:** Pre-computing utilities enables efficient candidate evaluation during genetic search

4. **Interpretable Weights:** Profile weights provide transparent representation of preference trade-offs

### Limitations

- Synthetic profiles may not perfectly represent actual population preferences
- Training requires substantial node feature data
- Model assumes preference independence across amenity types

---

## 6. Implementation

### File Structure

```
optimization-pipeline/
├── pnmlr_features.py    # Feature extraction (F=12)
├── pnmlr_profiles.py    # Dirichlet profile generation
├── pnmlr_model.py       # MLP architecture
├── pnmlr_hooks.py       # GA integration hooks
└── train_pnmlr.py       # Training script

models/
├── pnmlr_model.pkl      # Trained weights θ
├── pnmlr_profiles.json  # P×A weight matrix
├── pnmlr_normalizer.json# Feature normalization params
└── pnmlr_target_params.json
```

### Configuration

```yaml
pnmlr:
  enabled: true
  models_dir: "models"
  training:
    n_profiles: 20
    epochs: 100
    decay_constant: 2000.0
```

---

## References

1. Ponzi et al. (2025). PNMLR: Enhancing Route Recommendations With Personalized Preferences Using Graph Attention Networks.

2. McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.

3. Train, K. E. (2009). Discrete Choice Methods with Simulation.
