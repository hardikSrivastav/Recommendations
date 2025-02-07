import axios from 'axios';
import { Song } from '@/components/SongCard';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Function to get or create session ID
const getSessionId = () => {
  let sessionId = localStorage.getItem('session_id');
  if (!sessionId) {
    sessionId = `session_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('session_id', sessionId);
  }
  return sessionId;
};

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true
});

// Add session ID to every request
api.interceptors.request.use((config) => {
  config.headers['X-Session-ID'] = getSessionId();
  return config;
}, (error) => {
  return Promise.reject(error);
});

// Handle session ID from response
api.interceptors.response.use((response) => {
  const sessionId = response.headers['x-session-id'];
  if (sessionId) {
    localStorage.setItem('session_id', sessionId);
  }
  return response;
}, (error) => {
  return Promise.reject(error);
});

// Auth API
export const authAPI = {
  login: (credentials: { email: string; password: string }) => 
    api.post('/login', credentials),
  register: (userData: { email: string; password: string; name: string }) => 
    api.post('/register', userData),
};

// User API
export const userAPI = {
  updateProfile: (data: any) => 
    api.put('/users/profile', data),
  getDemographics: () => 
    api.get('/users/demographics'),
  updateDemographics: (data: any) => 
    api.put('/users/demographics', data),
  deleteUserData: () =>
    api.delete('/users/account'),
};

// Music API
export const musicAPI = {
  search: (query: string) => 
    api.get(`/music/search?q=${encodeURIComponent(query)}`),
  getDetails: (songId: number) => 
    api.get(`/music/songs/${songId}`),
};

// Recommendations API
export interface RecommendationResponse {
  user_id: string;
  timestamp: string;
  context: {
    total_songs: number;
    recent_songs: string[];
    demographics: Record<string, any>;
    is_cold_start: boolean;
  };
  metadata?: {
    requested_limit: number;
    buffer_limit: number;
    total_fetched: number;
  };
  predictions: Array<{
    song_id: string;
    confidence: number;
    predictor_weights: Record<string, number>;
    was_shown: boolean;
    was_selected: boolean;
  }>;
  songDetails?: Record<string, Song>;
}

export const recommendationsAPI = {
  get: () => 
    api.get<RecommendationResponse>('/recommendations'),
  getPersonalized: () => 
    api.get<RecommendationResponse>('/recommendations/personalized'),
  getTop5Suggestions: (limit: number = 5, useCache: boolean = false) =>
    api.get<RecommendationResponse>(`/recommendations/personalized?limit=${limit}&use_cache=${useCache}`),
  cachePredictions: (predictions: RecommendationResponse) =>
    api.post('/recommendations/cache', predictions),
  clearCache: () =>
    api.delete('/recommendations/cache'),
};

// Feedback API
export const feedbackAPI = {
  addToHistory: async (songId: number) => {
    try {
      console.log('Adding song to history:', songId);
      const response = await api.post('/music/history', { song_id: songId });
      console.log('Successfully added song to history:', response.data);
      return response;
    } catch (error: any) {
      console.error('Failed to add song to history:', error);
      console.error('Request details:', {
        songId,
        endpoint: '/music/history',
        error: error.response?.data || error.message
      });
      throw error;
    }
  },
  removeFromHistory: async (songId: number) => {
    try {
      console.log('Removing song from history:', songId);
      const response = await api.delete(`/music/history/${songId}`);
      console.log('Successfully removed song from history:', response.data);
      return response;
    } catch (error: any) {
      console.error('Failed to remove song from history:', error);
      console.error('Request details:', {
        songId,
        endpoint: `/music/history/${songId}`,
        error: error.response?.data || error.message
      });
      throw error;
    }
  },
  getHistory: async () => {
    try {
      console.log('Fetching history');
      const response = await api.get('/music/history');
      console.log('Successfully fetched history:', response.data);
      return response;
    } catch (error: any) {
      console.error('Failed to fetch history:', error);
      console.error('Request details:', {
        endpoint: '/music/history',
        error: error.response?.data || error.message
      });
      throw error;
    }
  },
};

export default api; 