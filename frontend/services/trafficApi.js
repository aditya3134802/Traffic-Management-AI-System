/**
 * Traffic API Service
 * 
 * This module provides client-side API functions for interacting with
 * the traffic management backend services.
 */

import axios from 'axios';

// Base API URL - configurable via environment
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
});

// Request interceptor for adding authentication
apiClient.interceptors.request.use(
  config => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  response => response,
  error => {
    const { response } = error;
    
    // Handle specific error codes
    if (response && response.status === 401) {
      // Authentication error - could trigger a login flow
      console.error('Authentication error');
    } else if (response && response.status === 403) {
      // Authorization error
      console.error('Authorization error: Insufficient permissions');
    } else if (response && response.status === 404) {
      console.error('Resource not found');
    } else if (response && response.status >= 500) {
      console.error('Server error occurred');
    } else if (!response) {
      console.error('Network error - check connection');
    }
    
    return Promise.reject(error);
  }
);

/**
 * Fetch traffic data based on specified parameters
 * 
 * @param {Object} params - Query parameters
 * @param {number} [params.startTime] - Start timestamp in milliseconds
 * @param {number} [params.endTime] - End timestamp in milliseconds
 * @param {string} [params.location] - Filter by location ID
 * @returns {Promise<Array>} - Promise resolving to array of traffic data points
 */
export const fetchTrafficData = async (params = {}) => {
  try {
    const response = await apiClient.get('/traffic-data', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching traffic data:', error);
    throw error;
  }
};

/**
 * Fetch anomalies based on specified parameters
 * 
 * @param {Object} params - Query parameters
 * @param {number} [params.startTime] - Start timestamp in milliseconds
 * @param {number} [params.endTime] - End timestamp in milliseconds
 * @param {string} [params.severity] - Filter by severity
 * @param {string} [params.type] - Filter by anomaly type
 * @param {string} [params.location] - Filter by location ID
 * @returns {Promise<Array>} - Promise resolving to array of anomalies
 */
export const fetchAnomalies = async (params = {}) => {
  try {
    const response = await apiClient.get('/anomalies', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching anomalies:', error);
    throw error;
  }
};

/**
 * Fetch anomaly statistics
 * 
 * @param {Object} params - Query parameters
 * @param {number} [params.startTime] - Start timestamp in milliseconds
 * @param {number} [params.endTime] - End timestamp in milliseconds
 * @returns {Promise<Object>} - Promise resolving to anomaly statistics
 */
export const fetchAnomalyStats = async (params = {}) => {
  try {
    const response = await apiClient.get('/anomaly/stats', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching anomaly statistics:', error);
    throw error;
  }
};

/**
 * Get AI analysis for a specific anomaly
 * 
 * @param {Object} anomaly - The anomaly to analyze
 * @returns {Promise<Object>} - Promise resolving to AI analysis results
 */
export const analyzeAnomaly = async (anomaly) => {
  try {
    if (!anomaly || !anomaly.id) {
      throw new Error('Invalid anomaly data');
    }
    
    const response = await apiClient.get(`/anomaly/analyze/${anomaly.id}`);
    return response.data;
  } catch (error) {
    console.error('Error analyzing anomaly:', error);
    throw error;
  }
};

/**
 * Fetch all available traffic monitoring locations
 * 
 * @returns {Promise<Array>} - Promise resolving to array of locations
 */
export const fetchLocations = async () => {
  try {
    const response = await apiClient.get('/locations');
    return response.data;
  } catch (error) {
    console.error('Error fetching locations:', error);
    throw error;
  }
};

/**
 * Trigger simulation data refresh (admin function)
 * 
 * @returns {Promise<Object>} - Promise resolving to refresh status
 */
export const refreshSimulation = async () => {
  try {
    const response = await apiClient.post('/simulate/refresh');
    return response.data;
  } catch (error) {
    console.error('Error refreshing simulation data:', error);
    throw error;
  }
};

/**
 * Export data to file (CSV, JSON)
 * 
 * @param {string} endpoint - API endpoint to export from
 * @param {Object} params - Query parameters
 * @param {string} format - Export format ('csv' or 'json')
 * @returns {Promise<Blob>} - Promise resolving to data blob
 */
export const exportData = async (endpoint, params = {}, format = 'csv') => {
  try {
    const response = await apiClient.get(`/export/${endpoint}`, {
      params: { ...params, format },
      responseType: 'blob'
    });
    
    return response.data;
  } catch (error) {
    console.error(`Error exporting ${endpoint} data:`, error);
    throw error;
  }
};

export default {
  fetchTrafficData,
  fetchAnomalies,
  fetchAnomalyStats,
  analyzeAnomaly,
  fetchLocations,
  refreshSimulation,
  exportData
};