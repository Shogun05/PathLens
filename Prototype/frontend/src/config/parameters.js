// Default optimization parameters
export const defaultParameters = {
  compositeWeights: {
    alpha: { value: 0.3, min: 0, max: 1, step: 0.05, label: 'Structure Weight (α)', description: 'Weight for network structure metrics' },
    beta: { value: 0.4, min: 0, max: 1, step: 0.05, label: 'Accessibility Weight (β)', description: 'Weight for amenity accessibility' },
    gamma: { value: 0.2, min: 0, max: 1, step: 0.05, label: 'Equity Weight (γ)', description: 'Weight for spatial equity metrics' },
    delta: { value: 0.1, min: 0, max: 1, step: 0.05, label: 'Travel Time Weight (δ)', description: 'Weight for travel time optimization' },
  },
  amenityAccessibility: {
    hospital: { value: 0.25, min: 0, max: 1, step: 0.05, label: 'Hospital', description: 'Importance of hospital accessibility' },
    school: { value: 0.2, min: 0, max: 1, step: 0.05, label: 'School', description: 'Importance of school accessibility' },
    grocery: { value: 0.15, min: 0, max: 1, step: 0.05, label: 'Grocery', description: 'Importance of grocery store accessibility' },
    transit: { value: 0.2, min: 0, max: 1, step: 0.05, label: 'Transit', description: 'Importance of public transit access' },
    park: { value: 0.1, min: 0, max: 1, step: 0.05, label: 'Park', description: 'Importance of park accessibility' },
    pharmacy: { value: 0.1, min: 0, max: 1, step: 0.05, label: 'Pharmacy', description: 'Importance of pharmacy accessibility' },
  },
  distanceThresholds: {
    maxWalkingDistance: { value: 1000, min: 200, max: 3000, step: 100, label: 'Max Walking Distance (m)', description: 'Maximum comfortable walking distance' },
    amenityCutoff: { value: 2000, min: 500, max: 5000, step: 100, label: 'Amenity Search Radius (m)', description: 'Search radius for nearby amenities' },
  },
  performance: {
    h3Resolution: { value: 9, min: 7, max: 11, step: 1, label: 'H3 Resolution', description: 'Hexagonal grid resolution (higher = finer detail)' },
    centralitySample: { value: 1000, min: 100, max: 5000, step: 100, label: 'Centrality Sample Size', description: 'Number of nodes to sample for centrality' },
    walkingSpeed: { value: 4.8, min: 3.0, max: 6.0, step: 0.1, label: 'Walking Speed (km/h)', description: 'Average walking speed for calculations' },
  },
};

// Parameter categories for organization
export const parameterCategories = [
  {
    key: 'compositeWeights',
    title: 'Composite Score Weights',
    description: 'Configure the weights for different components of the walkability score',
    icon: 'Scale',
  },
  {
    key: 'amenityAccessibility',
    title: 'Amenity Accessibility Weights',
    description: 'Set importance levels for different types of amenities',
    icon: 'Building2',
  },
  {
    key: 'distanceThresholds',
    title: 'Distance & Thresholds',
    description: 'Define distance parameters for walkability calculations',
    icon: 'Ruler',
  },
  {
    key: 'performance',
    title: 'Performance & Resolution',
    description: 'Tune performance parameters and calculation resolution',
    icon: 'Settings',
  },
];
