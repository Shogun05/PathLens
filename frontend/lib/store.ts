import { create } from 'zustand';

export interface Node {
  osmid: string;
  x: number;
  y: number;
  accessibility_score?: number;
  walkability_score?: number;
  equity_score?: number;
  travel_time_min?: number;
  betweenness_centrality?: number;
  dist_to_school?: number;
  dist_to_hospital?: number;
  dist_to_park?: number;
}

export interface Suggestion {
  type: string;
  geometry: {
    type: string;
    coordinates: [number, number];
  };
  properties: {
    id: string;
    amenity_type: string;
    expected_impact?: number;
    affected_nodes?: number;
    land_availability?: 'feasible' | 'limited' | 'unavailable';
    available_area_sqm?: number;
    satellite_confidence?: number;
    description?: string;
    impact_score?: number;
    cost_estimate?: string | number;
  };
}

interface PathLensState {
  // Location & Config
  location: string;
  customBounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  } | null;
  budget: number;
  maxAmenities: number;
  addSchools: boolean;
  addHospitals: boolean;
  addParks: boolean;
  addBusStations: boolean;
  demoMode: boolean;

  // Data
  baselineNodes: Node[];
  optimizedNodes: Node[];
  suggestions: Suggestion[];
  selectedSuggestionIds: Set<string>;

  // Loading states
  isOptimizing: boolean;
  isRescoring: boolean;
  optimizationProgress: string;

  // Metrics
  baselineScore: number;
  optimizedScore: number;
  currentScore: number;

  // Actions
  setLocation: (location: string) => void;
  setCustomBounds: (bounds: { north: number; south: number; east: number; west: number; } | null) => void;
  setBudget: (budget: number) => void;
  setMaxAmenities: (max: number) => void;
  setAddSchools: (add: boolean) => void;
  setAddHospitals: (add: boolean) => void;
  setAddParks: (add: boolean) => void;
  setAddBusStations: (add: boolean) => void;
  setDemoMode: (demoMode: boolean) => void;
  setBaselineNodes: (nodes: Node[]) => void;
  setOptimizedNodes: (nodes: Node[]) => void;
  setSuggestions: (suggestions: Suggestion[]) => void;
  toggleSuggestion: (id: string) => void;
  setIsOptimizing: (optimizing: boolean) => void;
  setIsRescoring: (rescoring: boolean) => void;
  setOptimizationProgress: (progress: string) => void;
  setBaselineScore: (score: number) => void;
  setOptimizedScore: (score: number) => void;
  setCurrentScore: (score: number) => void;
  reset: () => void;
}

export const usePathLensStore = create<PathLensState>((set) => ({
  // Initial state
  location: '',
  customBounds: null,
  budget: 50000000,
  maxAmenities: 10,
  addSchools: true,
  addHospitals: true,
  addParks: true,
  addBusStations: false,
  baselineNodes: [],
  optimizedNodes: [],
  suggestions: [],
  selectedSuggestionIds: new Set(),
  isOptimizing: false,
  isRescoring: false,
  optimizationProgress: '',
  baselineScore: 0,
  optimizedScore: 0,
  currentScore: 0,
  demoMode: false,

  // Actions
  setLocation: (location) => set({ location }),
  setCustomBounds: (customBounds) => set({ customBounds }),
  setBudget: (budget) => set({ budget }),
  setMaxAmenities: (maxAmenities) => set({ maxAmenities }),
  setAddSchools: (addSchools) => set({ addSchools }),
  setAddHospitals: (addHospitals) => set({ addHospitals }),
  setAddParks: (addParks) => set({ addParks }),
  setAddBusStations: (addBusStations) => set({ addBusStations }),
  setDemoMode: (demoMode) => set({ demoMode }),
  setBaselineNodes: (baselineNodes) => set({ baselineNodes }),
  setOptimizedNodes: (optimizedNodes) => set({ optimizedNodes }),
  setSuggestions: (suggestions) => set({ suggestions }),
  toggleSuggestion: (id) =>
    set((state) => {
      const newSet = new Set(state.selectedSuggestionIds);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return { selectedSuggestionIds: newSet };
    }),
  setIsOptimizing: (isOptimizing) => set({ isOptimizing }),
  setIsRescoring: (isRescoring) => set({ isRescoring }),
  setOptimizationProgress: (optimizationProgress) => set({ optimizationProgress }),
  setBaselineScore: (baselineScore) => set({ baselineScore }),
  setOptimizedScore: (optimizedScore) => set({ optimizedScore }),
  setCurrentScore: (currentScore) => set({ currentScore }),
  reset: () =>
    set({
      baselineNodes: [],
      optimizedNodes: [],
      suggestions: [],
      selectedSuggestionIds: new Set(),
      isOptimizing: false,
      isRescoring: false,
      optimizationProgress: '',
      baselineScore: 0,
      optimizedScore: 0,
      currentScore: 0,
    }),
}));
