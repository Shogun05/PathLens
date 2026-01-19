import axios from 'axios';
import type { Node, Suggestion } from './store';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  validateStatus: (status) => status < 500, // Don't throw on 4xx errors
});

// Add response interceptor to handle empty responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle network errors
    if (!error.response) {
      console.error('Network error - backend may not be running:', error.message);
      return Promise.reject(new Error('Backend server is not responding. Please ensure it is running.'));
    }

    // Handle empty response bodies
    if (error.response && !error.response.data) {
      console.error('Empty response from backend:', error.response.status);
      error.response.data = { error: 'Empty response from server' };
    }

    return Promise.reject(error);
  }
);

export interface OptimizeRequest {
  location: string;
  budget: number;
  max_amenities: number;
  add_schools: boolean;
  add_hospitals: boolean;
  add_parks: boolean;
}

export interface RescoreRequest {
  location: string;
  selected_ids: string[];
}

export interface NodesQueryParams {
  type: 'baseline' | 'optimized';
  city?: string;
  mode?: 'ga_only' | 'ga_milp' | 'ga_milp_pnmlr';
  limit?: number;
  offset?: number;
  bbox?: string; // "west,south,east,north"
}

export interface OptimizationStatus {
  status: string;
  stage?: string;
  message?: string;
  percent?: number;
  timestamp?: string;
  pipelines?: Record<string, string>;
  details?: Record<string, unknown>;
  error?: string;
}

export const pathLensAPI = {
  optimize: async (data: OptimizeRequest) => {
    const response = await api.post('/api/optimize', data);
    return response.data;
  },

  getNodes: async (params: NodesQueryParams | 'baseline' | 'optimized'): Promise<Node[]> => {
    // Support both old signature (string) and new signature (params object)
    if (typeof params === 'string') {
      params = { type: params };
    }

    // Auto-include city from sessionStorage if not provided
    const city = params.city || (typeof window !== 'undefined' ? sessionStorage.getItem('selectedCity') : null);
    // Default optimization mode depends on city data availability
    // Bangalore: ga_milp_pnmlr (has +0.8 delta)
    // Navi Mumbai / Mumbai: ga_only (verified working)
    let defaultMode = 'ga_milp_pnmlr';
    if (city === 'navi_mumbai' || city === 'mumbai') {
      defaultMode = 'ga_only';
    }
    const mode = params.mode || (params.type === 'optimized' ? defaultMode : undefined);

    const queryParams = new URLSearchParams();
    queryParams.append('type', params.type);
    if (city) queryParams.append('city', city);
    if (mode) queryParams.append('mode', mode);
    if (params.limit) queryParams.append('limit', params.limit.toString());
    if (params.offset) queryParams.append('offset', params.offset.toString());
    if (params.bbox) queryParams.append('bbox', params.bbox);

    const response = await api.get(`/api/nodes?${queryParams.toString()}`);
    return response.data;
  },

  getSuggestions: async (city?: string, mode?: string): Promise<{ type: string; features: Suggestion[] }> => {
    // Use provided city, or check sessionStorage, or fall back to default
    const selectedCity = city || (typeof window !== 'undefined' ? sessionStorage.getItem('selectedCity') : null) || 'bangalore';

    // Default mode based on city
    let defaultMode = 'ga_milp_pnmlr';
    if (selectedCity === 'navi_mumbai' || selectedCity === 'mumbai') {
      defaultMode = 'ga_only';
    }
    const selectedMode = mode || defaultMode;

    const response = await api.get(`/api/suggestions?city=${selectedCity}&mode=${selectedMode}`);
    return response.data;
  },

  getMetricsSummary: async (type: 'baseline' | 'optimized', city?: string, mode?: string) => {
    // Use provided city, or check sessionStorage, or fall back to default
    const selectedCity = city || (typeof window !== 'undefined' ? sessionStorage.getItem('selectedCity') : null) || 'bangalore';

    // Default optimization mode depends on city data availability
    let defaultMode = 'ga_milp_pnmlr';
    if (selectedCity === 'navi_mumbai' || selectedCity === 'mumbai') {
      defaultMode = 'ga_only';
    }

    // Default mode for optimized is dynamic, for baseline show ga_only
    const selectedMode = mode || (type === 'optimized' ? defaultMode : 'ga_only');
    const response = await api.get(`/api/metrics-summary?type=${type}&city=${selectedCity}&mode=${selectedMode}`);
    return response.data;
  },

  rescore: async (data: RescoreRequest) => {
    const response = await api.post('/api/rescore', data);
    return response.data;
  },

  // Optimization endpoints
  getOptimizationStatus: async (): Promise<OptimizationStatus> => {
    const response = await api.get('/api/optimization/status');
    return response.data;
  },

  getOptimizationResults: async (city?: string, mode?: string) => {
    // Use provided city, or check sessionStorage, or fall back to bangalore
    const selectedCity = city || (typeof window !== 'undefined' ? sessionStorage.getItem('selectedCity') : null) || 'bangalore';

    // Default mode based on city
    let defaultMode = 'ga_milp_pnmlr';
    if (selectedCity === 'navi_mumbai' || selectedCity === 'mumbai') {
      defaultMode = 'ga_only';
    }
    const selectedMode = mode || defaultMode;

    const response = await api.get(`/api/optimization/results?city=${selectedCity}&mode=${selectedMode}`);
    return response.data;
  },

  getOptimizationPois: async (city?: string, mode?: string) => {
    // Use provided city, or check sessionStorage, or fall back to bangalore
    const selectedCity = city || (typeof window !== 'undefined' ? sessionStorage.getItem('selectedCity') : null) || 'bangalore';

    // Default mode based on city
    let defaultMode = 'ga_milp_pnmlr';
    if (selectedCity === 'navi_mumbai' || selectedCity === 'mumbai') {
      defaultMode = 'ga_only';
    }
    const selectedMode = mode || defaultMode;

    // Always use city-specific endpoint (no legacy fallback)
    const response = await api.get(`/api/modes/${selectedMode}/pois?city=${selectedCity}`);
    return response.data;
  },

  getOptimizationHistory: async () => {
    const response = await api.get('/api/optimization/history');
    return response.data;
  },

  getOptimizationSummary: async () => {
    const response = await api.get('/api/optimization/summary');
    return response.data;
  },

  getOptimizationComparison: async () => {
    const response = await api.get('/api/optimization/comparison');
    return response.data;
  },

  // City data status check
  getCityDataStatus: async (city: string) => {
    const response = await api.get(`/api/city-data-status?city=${city}`);
    return response.data;
  },

  // Multi-mode optimization endpoints
  getModes: async (city: string = 'bangalore') => {
    const response = await api.get(`/api/modes?city=${city}`);
    return response.data;
  },

  getModeNodes: async (mode: string, city: string = 'bangalore', limit: number = 5000) => {
    const response = await api.get(`/api/modes/${mode}/nodes?city=${city}&limit=${limit}`);
    return response.data;
  },

  getModeMetrics: async (mode: string, city: string = 'bangalore') => {
    const response = await api.get(`/api/modes/${mode}/metrics?city=${city}`);
    return response.data;
  },

  getModePois: async (mode: string, city: string = 'bangalore') => {
    const response = await api.get(`/api/modes/${mode}/pois?city=${city}`);
    return response.data;
  },

  getModesComparison: async (city: string = 'bangalore') => {
    const response = await api.get(`/api/modes/comparison?city=${city}`);
    return response.data;
  },
};
