export const generateDemoNodes = (center: [number, number], count: number) => {
  return Array.from({ length: count }).map((_, i) => ({
    osmid: `demo-${i}`,
    y: center[0] + (Math.random() - 0.5) * 0.05,
    x: center[1] + (Math.random() - 0.5) * 0.05,
    score: Math.floor(Math.random() * 100),
    type: 'intersection',
    accessibility_score: Math.floor(Math.random() * 100),
    walkability_score: Math.floor(Math.random() * 100),
    equity_score: Math.floor(Math.random() * 100),
    travel_time_min: Math.floor(Math.random() * 60),
    dist_to_park: Math.floor(Math.random() * 1000),
  }));
};
