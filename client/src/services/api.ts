import axios from 'axios';

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
};

// Music API
export const musicAPI = {
  search: (query: string) => 
    api.get(`/music/search?q=${encodeURIComponent(query)}`),
  getDetails: (songId: number) => 
    api.get(`/music/${songId}`),
};

// Recommendations API
export const recommendationsAPI = {
  get: () => 
    api.get('/recommendations'),
  getPersonalized: () => 
    api.get('/recommendations/personalized'),
};

// Feedback API
export const feedbackAPI = {
  addToHistory: (songId: number) => 
    api.post('/feedback/history', { song_id: songId }),
  removeFromHistory: (songId: number) => 
    api.delete(`/feedback/history/${songId}`),
  getHistory: () => 
    api.get('/feedback/history'),
};

export default api; 