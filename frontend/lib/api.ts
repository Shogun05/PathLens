import axios from 'axios';
import type { Node, Suggestion } from './store';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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
  limit?: number;
  offset?: number;
  bbox?: string; // "west,south,east,north"
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
    
    const queryParams = new URLSearchParams();
    queryParams.append('type', params.type);
    if (params.limit) queryParams.append('limit', params.limit.toString());
    if (params.offset) queryParams.append('offset', params.offset.toString());
    if (params.bbox) queryParams.append('bbox', params.bbox);
    
    const response = await api.get(`/api/nodes?${queryParams.toString()}`);
    return response.data;
  },

  getSuggestions: async (): Promise<{ type: string; features: Suggestion[] }> => {
    const response = await api.get('/api/suggestions');
    return response.data;
  },

  getMetricsSummary: async (type: 'baseline' | 'optimized') => {
    const response = await api.get(`/api/metrics-summary?type=${type}`);
    return response.data;
  },

  rescore: async (data: RescoreRequest) => {
    const response = await api.post('/api/rescore', data);
    return response.data;
  },

  // Optimization endpoints
  getOptimizationResults: async () => {
    const response = await api.get('/api/optimization/results');
    return response.data;
  },

  getOptimizationPois: async () => {
    const response = await api.get('/api/optimization/pois');
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
};
