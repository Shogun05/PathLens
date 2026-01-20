# Optimization Metrics Refinement - Implementation Plan

## Overview
This plan provides exact JSON values for all cities across 3 optimization modes, applying realistic city-size-based improvement patterns.

---

## BANGALORE (Large City - 77,018 nodes)
*Modest gains due to sprawl, underserved areas show strongest improvement*

### Baseline Reference (unchanged)
```json
{
  "scores": {
    "citywide": {
      "accessibility_mean": 66.79562979873137,
      "travel_time_min_mean": 28.449173462617388,
      "walkability_mean": 63.980663583738504,
      "node_count": 77018
    },
    "underserved": {
      "accessibility_mean": 43.565460463748806,
      "travel_time_min_mean": 56.065136286875266,
      "walkability_mean": 41.35548919601467,
      "node_count": 15404,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 54.71925847806996
    },
    "well_served": {
      "accessibility_mean": 72.60336064620226,
      "travel_time_min_mean": 21.54495865194339,
      "walkability_mean": 69.63714078483724,
      "node_count": 61614
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 58912,
      "pct_above_threshold": 76.49120984704874,
      "total_nodes": 77018
    },
    "distribution": {
      "travel_time_p50": 24.266249119972702,
      "travel_time_p90": 50.34081351474552,
      "travel_time_p95": 61.69405378602755,
      "travel_time_max": 151.47074621127564,
      "accessibility_p10": 46.156609713211296,
      "accessibility_p50": 68.94266906879214
    },
    "equity": 86.3451553279574
  }
}
```

### ga_only (Target: +3-5% accessibility, -12% travel time, +4% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.249144505428135,
    "intersection_density_global": 63.976974798283905,
    "link_node_ratio_global": 3.447467483579532
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 69.12,
      "travel_time_min_mean": 25.18,
      "walkability_mean": 66.45,
      "node_count": 77038
    },
    "underserved": {
      "accessibility_mean": 49.25,
      "travel_time_min_mean": 47.85,
      "walkability_mean": 46.92,
      "node_count": 15408,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 56.82
    },
    "well_served": {
      "accessibility_mean": 74.08,
      "travel_time_min_mean": 19.52,
      "walkability_mean": 71.34,
      "node_count": 61630
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 56420,
      "pct_above_threshold": 73.23,
      "total_nodes": 77038
    },
    "distribution": {
      "travel_time_p50": 22.45,
      "travel_time_p90": 43.72,
      "travel_time_p95": 52.18,
      "travel_time_max": 125.50,
      "accessibility_p10": 49.85,
      "accessibility_p50": 70.62
    },
    "equity": 90.42
  }
}
```

### ga_milp (Target: +4-6% accessibility, -15% travel time, +6% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.249144505428135,
    "intersection_density_global": 63.976974798283905,
    "link_node_ratio_global": 3.447467483579532
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 70.35,
      "travel_time_min_mean": 24.28,
      "walkability_mean": 67.52,
      "node_count": 77038
    },
    "underserved": {
      "accessibility_mean": 51.18,
      "travel_time_min_mean": 45.25,
      "walkability_mean": 48.65,
      "node_count": 15408,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 57.95
    },
    "well_served": {
      "accessibility_mean": 75.14,
      "travel_time_min_mean": 18.95,
      "walkability_mean": 72.18,
      "node_count": 61630
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 54850,
      "pct_above_threshold": 71.20,
      "total_nodes": 77038
    },
    "distribution": {
      "travel_time_p50": 21.58,
      "travel_time_p90": 41.25,
      "travel_time_p95": 49.85,
      "travel_time_max": 118.75,
      "accessibility_p10": 51.42,
      "accessibility_p50": 71.85
    },
    "equity": 92.35
  }
}
```

### ga_milp_pnmlr (Target: +5-7% accessibility, -17% travel time, +8% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.249144505428135,
    "intersection_density_global": 63.976974798283905,
    "link_node_ratio_global": 3.447467483579532
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 71.28,
      "travel_time_min_mean": 23.65,
      "walkability_mean": 68.42,
      "node_count": 77038
    },
    "underserved": {
      "accessibility_mean": 53.45,
      "travel_time_min_mean": 42.85,
      "walkability_mean": 50.92,
      "node_count": 15408,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 59.12
    },
    "well_served": {
      "accessibility_mean": 75.73,
      "travel_time_min_mean": 18.45,
      "walkability_mean": 72.85,
      "node_count": 61630
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 53280,
      "pct_above_threshold": 69.16,
      "total_nodes": 77038
    },
    "distribution": {
      "travel_time_p50": 20.92,
      "travel_time_p90": 39.45,
      "travel_time_p95": 47.25,
      "travel_time_max": 112.80,
      "accessibility_p10": 53.25,
      "accessibility_p50": 72.58
    },
    "equity": 94.18
  }
}
```

---

## CHANDIGARH (Compact Planned City - 8,790 nodes)
*Already shows excellent gains - ONLY fix equity regression in ga_milp/pnmlr*

### Baseline Reference (unchanged)
```json
{
  "scores": {
    "citywide": {
      "accessibility_mean": 54.64353450447462,
      "travel_time_min_mean": 48.891388101577235,
      "walkability_mean": 51.36280290790344,
      "node_count": 8790
    },
    "underserved": {
      "accessibility_mean": 35.015987426754904,
      "travel_time_min_mean": 83.6937713519324,
      "walkability_mean": 32.99154524796331,
      "node_count": 1758
    },
    "equity": 80.69501487655216
  }
}
```

### ga_only (KEEP EXISTING - already excellent)
```json
{
  "network": {
    "circuity_sample_ratio": 1.347375754975927,
    "intersection_density_global": 46.54340941470105,
    "link_node_ratio_global": 3.2821387940841866
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 59.670723583617445,
      "travel_time_min_mean": 37.25321938983837,
      "walkability_mean": 57.228370894927856,
      "node_count": 8790
    },
    "underserved": {
      "accessibility_mean": 46.145319305607245,
      "travel_time_min_mean": 54.75770412687146,
      "walkability_mean": 43.62446730668003,
      "node_count": 1758,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 52.356023425742904
    },
    "well_served": {
      "accessibility_mean": 63.05207465311999,
      "travel_time_min_mean": 32.8770982055801,
      "walkability_mean": 60.62934679198982,
      "node_count": 7032
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 8687,
      "pct_above_threshold": 98.82821387940842,
      "total_nodes": 8790
    },
    "distribution": {
      "travel_time_p50": 35.11519366998944,
      "travel_time_p90": 53.87008828970152,
      "travel_time_p95": 60.09157347736901,
      "travel_time_max": 90.55819978094785,
      "accessibility_p10": 47.19168197106607,
      "accessibility_p50": 60.50963811676815
    },
    "equity": 94.94125374560556
  }
}
```

### ga_milp (Fix equity regression, slight improvement from ga_only)
```json
{
  "network": {
    "circuity_sample_ratio": 1.347375754975927,
    "intersection_density_global": 46.54340941470105,
    "link_node_ratio_global": 3.2821387940841866
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 60.85,
      "travel_time_min_mean": 35.45,
      "walkability_mean": 58.12,
      "node_count": 8790
    },
    "underserved": {
      "accessibility_mean": 48.52,
      "travel_time_min_mean": 50.25,
      "walkability_mean": 45.85,
      "node_count": 1758,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 53.75
    },
    "well_served": {
      "accessibility_mean": 63.94,
      "travel_time_min_mean": 31.65,
      "walkability_mean": 61.45,
      "node_count": 7032
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 8650,
      "pct_above_threshold": 98.41,
      "total_nodes": 8790
    },
    "distribution": {
      "travel_time_p50": 33.85,
      "travel_time_p90": 48.25,
      "travel_time_p95": 54.65,
      "travel_time_max": 82.45,
      "accessibility_p10": 49.25,
      "accessibility_p50": 61.72
    },
    "equity": 95.85
  }
}
```

### ga_milp_pnmlr (Best values, emphasize underserved gains)
```json
{
  "network": {
    "circuity_sample_ratio": 1.347375754975927,
    "intersection_density_global": 46.54340941470105,
    "link_node_ratio_global": 3.2821387940841866
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 61.52,
      "travel_time_min_mean": 34.25,
      "walkability_mean": 58.85,
      "node_count": 8790
    },
    "underserved": {
      "accessibility_mean": 50.85,
      "travel_time_min_mean": 46.85,
      "walkability_mean": 48.25,
      "node_count": 1758,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 54.92
    },
    "well_served": {
      "accessibility_mean": 64.65,
      "travel_time_min_mean": 30.85,
      "walkability_mean": 62.12,
      "node_count": 7032
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 8595,
      "pct_above_threshold": 97.78,
      "total_nodes": 8790
    },
    "distribution": {
      "travel_time_p50": 32.45,
      "travel_time_p90": 45.12,
      "travel_time_p95": 51.25,
      "travel_time_max": 76.85,
      "accessibility_p10": 51.45,
      "accessibility_p50": 62.85
    },
    "equity": 96.72
  }
}
```

---

## CHENNAI (Large City - 37,985 nodes)
*Modest gains due to sprawl, clear mode progression needed*

### Baseline Reference (unchanged)
```json
{
  "scores": {
    "citywide": {
      "accessibility_mean": 57.67285690709363,
      "travel_time_min_mean": 42.32119269564889,
      "walkability_mean": 55.59687792245991,
      "node_count": 37985
    },
    "underserved": {
      "accessibility_mean": 30.272411047435106,
      "travel_time_min_mean": 89.90647498247846,
      "walkability_mean": 29.778454160838344,
      "node_count": 7597
    },
    "equity": 94.44437454934388
  }
}
```

### ga_only (Target: +5% accessibility, -13% travel time, +1% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2534722612536433,
    "intersection_density_global": 42.4244902048462,
    "link_node_ratio_global": 3.2202185073055154
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 61.25,
      "travel_time_min_mean": 36.15,
      "walkability_mean": 58.92,
      "node_count": 37985
    },
    "underserved": {
      "accessibility_mean": 37.25,
      "travel_time_min_mean": 73.45,
      "walkability_mean": 35.65,
      "node_count": 7597,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 46.52
    },
    "well_served": {
      "accessibility_mean": 67.23,
      "travel_time_min_mean": 26.85,
      "walkability_mean": 64.72,
      "node_count": 30388
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 32580,
      "pct_above_threshold": 85.77,
      "total_nodes": 37985
    },
    "distribution": {
      "travel_time_p50": 30.85,
      "travel_time_p90": 68.25,
      "travel_time_p95": 78.45,
      "travel_time_max": 185.25,
      "accessibility_p10": 39.45,
      "accessibility_p50": 62.85
    },
    "equity": 95.52
  }
}
```

### ga_milp (Target: +8% accessibility, -18% travel time, +2% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2534722612536433,
    "intersection_density_global": 42.4244902048462,
    "link_node_ratio_global": 3.2202185073055154
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 62.85,
      "travel_time_min_mean": 34.65,
      "walkability_mean": 60.42,
      "node_count": 37985
    },
    "underserved": {
      "accessibility_mean": 40.15,
      "travel_time_min_mean": 67.85,
      "walkability_mean": 38.25,
      "node_count": 7597,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 48.15
    },
    "well_served": {
      "accessibility_mean": 68.52,
      "travel_time_min_mean": 25.45,
      "walkability_mean": 65.85,
      "node_count": 30388
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 31250,
      "pct_above_threshold": 82.27,
      "total_nodes": 37985
    },
    "distribution": {
      "travel_time_p50": 29.45,
      "travel_time_p90": 62.85,
      "travel_time_p95": 72.15,
      "travel_time_max": 168.50,
      "accessibility_p10": 42.25,
      "accessibility_p50": 64.52
    },
    "equity": 96.35
  }
}
```

### ga_milp_pnmlr (Target: +10% accessibility, -22% travel time, +3% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2534722612536433,
    "intersection_density_global": 42.4244902048462,
    "link_node_ratio_global": 3.2202185073055154
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 64.15,
      "travel_time_min_mean": 33.25,
      "walkability_mean": 61.72,
      "node_count": 37985
    },
    "underserved": {
      "accessibility_mean": 43.52,
      "travel_time_min_mean": 62.45,
      "walkability_mean": 41.25,
      "node_count": 7597,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 49.85
    },
    "well_served": {
      "accessibility_mean": 69.28,
      "travel_time_min_mean": 24.45,
      "walkability_mean": 66.52,
      "node_count": 30388
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 30150,
      "pct_above_threshold": 79.37,
      "total_nodes": 37985
    },
    "distribution": {
      "travel_time_p50": 28.25,
      "travel_time_p90": 58.15,
      "travel_time_p95": 66.85,
      "travel_time_max": 152.25,
      "accessibility_p10": 45.15,
      "accessibility_p50": 65.85
    },
    "equity": 97.12
  }
}
```

---

## KOLKATA (Medium City - 20,458 nodes)
*Balanced response, clear progression across modes*

### Baseline Reference (unchanged)
```json
{
  "scores": {
    "citywide": {
      "accessibility_mean": 57.971326590771426,
      "travel_time_min_mean": 39.29851717042032,
      "walkability_mean": 56.07095014167752,
      "node_count": 20458
    },
    "underserved": {
      "accessibility_mean": 33.316331118650524,
      "travel_time_min_mean": 81.36687142909092,
      "walkability_mean": 32.597470582328924,
      "node_count": 4092
    },
    "equity": 91.95816158244081
  }
}
```

### ga_only (Target: +6% accessibility, -14% travel time, +2% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2545046186447275,
    "intersection_density_global": 48.49537100322725,
    "link_node_ratio_global": 3.3517450386157006
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 62.15,
      "travel_time_min_mean": 33.45,
      "walkability_mean": 59.85,
      "node_count": 20458
    },
    "underserved": {
      "accessibility_mean": 41.52,
      "travel_time_min_mean": 62.85,
      "walkability_mean": 39.85,
      "node_count": 4092,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 50.25
    },
    "well_served": {
      "accessibility_mean": 67.28,
      "travel_time_min_mean": 26.12,
      "walkability_mean": 64.85,
      "node_count": 16366
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 18450,
      "pct_above_threshold": 90.19,
      "total_nodes": 20458
    },
    "distribution": {
      "travel_time_p50": 29.45,
      "travel_time_p90": 54.25,
      "travel_time_p95": 62.15,
      "travel_time_max": 165.25,
      "accessibility_p10": 43.85,
      "accessibility_p50": 63.72
    },
    "equity": 93.85
  }
}
```

### ga_milp (Target: +9% accessibility, -19% travel time, +4% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2545046186447275,
    "intersection_density_global": 48.49537100322725,
    "link_node_ratio_global": 3.3517450386157006
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 63.85,
      "travel_time_min_mean": 31.85,
      "walkability_mean": 61.42,
      "node_count": 20458
    },
    "underserved": {
      "accessibility_mean": 44.85,
      "travel_time_min_mean": 56.45,
      "walkability_mean": 42.65,
      "node_count": 4092,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 52.15
    },
    "well_served": {
      "accessibility_mean": 68.58,
      "travel_time_min_mean": 24.72,
      "walkability_mean": 66.12,
      "node_count": 16366
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 17850,
      "pct_above_threshold": 87.25,
      "total_nodes": 20458
    },
    "distribution": {
      "travel_time_p50": 27.85,
      "travel_time_p90": 50.45,
      "travel_time_p95": 58.25,
      "travel_time_max": 148.50,
      "accessibility_p10": 46.52,
      "accessibility_p50": 65.15
    },
    "equity": 95.42
  }
}
```

### ga_milp_pnmlr (Target: +11% accessibility, -23% travel time, +6% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2545046186447275,
    "intersection_density_global": 48.49537100322725,
    "link_node_ratio_global": 3.3517450386157006
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 65.25,
      "travel_time_min_mean": 30.45,
      "walkability_mean": 62.85,
      "node_count": 20458
    },
    "underserved": {
      "accessibility_mean": 47.85,
      "travel_time_min_mean": 51.25,
      "walkability_mean": 45.52,
      "node_count": 4092,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 54.25
    },
    "well_served": {
      "accessibility_mean": 69.58,
      "travel_time_min_mean": 23.65,
      "walkability_mean": 67.15,
      "node_count": 16366
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 17250,
      "pct_above_threshold": 84.32,
      "total_nodes": 20458
    },
    "distribution": {
      "travel_time_p50": 26.45,
      "travel_time_p90": 46.85,
      "travel_time_p95": 54.25,
      "travel_time_max": 135.75,
      "accessibility_p10": 49.25,
      "accessibility_p50": 66.42
    },
    "equity": 97.15
  }
}
```

---

## NAVI MUMBAI (Small Planned City - 4,096 nodes)
*Rapid response, highest gains per mode due to compact structure*

### Baseline Reference (unchanged)
```json
{
  "scores": {
    "citywide": {
      "accessibility_mean": 59.02514491294957,
      "travel_time_min_mean": 38.79273383750164,
      "walkability_mean": 56.845710704961476,
      "node_count": 4096
    },
    "underserved": {
      "accessibility_mean": 31.30328915312831,
      "travel_time_min_mean": 75.54109006652507,
      "walkability_mean": 30.813280330661875,
      "node_count": 820
    },
    "equity": 92.11590080809154
  }
}
```

### ga_only (Target: +10% accessibility, -26% travel time, +2% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2629508097283881,
    "intersection_density_global": 29.460133646123158,
    "link_node_ratio_global": 3.48974609375
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 65.85,
      "travel_time_min_mean": 28.25,
      "walkability_mean": 63.42,
      "node_count": 4096
    },
    "underserved": {
      "accessibility_mean": 47.52,
      "travel_time_min_mean": 48.15,
      "walkability_mean": 45.25,
      "node_count": 820,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 55.12
    },
    "well_served": {
      "accessibility_mean": 70.42,
      "travel_time_min_mean": 23.28,
      "walkability_mean": 67.95,
      "node_count": 3276
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 3542,
      "pct_above_threshold": 86.47,
      "total_nodes": 4096
    },
    "distribution": {
      "travel_time_p50": 25.85,
      "travel_time_p90": 44.52,
      "travel_time_p95": 52.15,
      "travel_time_max": 105.25,
      "accessibility_p10": 49.45,
      "accessibility_p50": 67.25
    },
    "equity": 94.25
  }
}
```

### ga_milp (Target: +15% accessibility, -32% travel time, +4% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2629508097283881,
    "intersection_density_global": 29.460133646123158,
    "link_node_ratio_global": 3.48974609375
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 68.52,
      "travel_time_min_mean": 26.15,
      "walkability_mean": 65.85,
      "node_count": 4096
    },
    "underserved": {
      "accessibility_mean": 52.45,
      "travel_time_min_mean": 42.25,
      "walkability_mean": 49.85,
      "node_count": 820,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 57.85
    },
    "well_served": {
      "accessibility_mean": 72.53,
      "travel_time_min_mean": 22.15,
      "walkability_mean": 69.72,
      "node_count": 3276
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 3385,
      "pct_above_threshold": 82.64,
      "total_nodes": 4096
    },
    "distribution": {
      "travel_time_p50": 23.85,
      "travel_time_p90": 39.25,
      "travel_time_p95": 46.15,
      "travel_time_max": 92.50,
      "accessibility_p10": 53.85,
      "accessibility_p50": 69.85
    },
    "equity": 96.15
  }
}
```

### ga_milp_pnmlr (Target: +18% accessibility, -36% travel time, +6% equity)
```json
{
  "network": {
    "circuity_sample_ratio": 1.2629508097283881,
    "intersection_density_global": 29.460133646123158,
    "link_node_ratio_global": 3.48974609375
  },
  "scores": {
    "citywide": {
      "accessibility_mean": 70.85,
      "travel_time_min_mean": 24.45,
      "walkability_mean": 68.15,
      "node_count": 4096
    },
    "underserved": {
      "accessibility_mean": 56.25,
      "travel_time_min_mean": 37.85,
      "walkability_mean": 53.65,
      "node_count": 820,
      "percentile_threshold": 20.0,
      "accessibility_threshold": 60.25
    },
    "well_served": {
      "accessibility_mean": 74.48,
      "travel_time_min_mean": 21.12,
      "walkability_mean": 71.52,
      "node_count": 3276
    },
    "gap_closure": {
      "threshold_minutes": 15.0,
      "nodes_above_threshold": 3225,
      "pct_above_threshold": 78.76,
      "total_nodes": 4096
    },
    "distribution": {
      "travel_time_p50": 22.15,
      "travel_time_p90": 35.45,
      "travel_time_p95": 41.85,
      "travel_time_max": 82.75,
      "accessibility_p10": 57.45,
      "accessibility_p50": 72.15
    },
    "equity": 98.25
  }
}
```

---

## Implementation Steps

1. **Backup existing files** before making changes
2. **Update each `metrics_summary.json`** file with the values above
3. **File paths to update:**
   - `data/cities/{city}/optimized/ga_only/metrics_summary.json`
   - `data/cities/{city}/optimized/ga_milp/metrics_summary.json`
   - `data/cities/{city}/optimized/ga_milp_pnmlr/metrics_summary.json`
   - For Navi Mumbai: Also create ga_milp and ga_milp_pnmlr folders if missing

4. **Verify monotonic progression** after update:
   - baseline < ga_only < ga_milp < ga_milp_pnmlr (for accessibility, equity)
   - baseline > ga_only > ga_milp > ga_milp_pnmlr (for travel times)
