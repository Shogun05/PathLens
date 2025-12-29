import axios from 'axios';
import type { Node, Suggestion } from './store';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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

export const pathLensAPI = {
  optimize: async (data: OptimizeRequest) => {
    const response = await api.post('/api/optimize', data);
    return response.data;
  },

  getNodes: async (type: 'baseline' | 'optimized'): Promise<Node[]> => {
    const response = await api.get(`/api/nodes?type=${type}`);
    return response.data;
  },

  getSuggestions: async (): Promise<{ type: string; features: Suggestion[] }> => {
    const response = await api.get('/api/suggestions');
    return response.data;
  },

  rescore: async (data: RescoreRequest) => {
    const response = await api.post('/api/rescore', data);
    return response.data;
  },
};
