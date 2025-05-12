import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { 
  Box, 
  Flex, 
  Text, 
  Heading, 
  Select, 
  FormControl, 
  FormLabel, 
  Stat, 
  StatLabel, 
  StatNumber, 
  StatHelpText, 
  Badge, 
  Spinner, 
  Button, 
  useColorModeValue,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  CloseButton,
  Grid,
  GridItem,
  Tooltip,
  Icon,
  useDisclosure 
} from '@chakra-ui/react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer,
  Area,
  AreaChart,
  ScatterChart,
  Scatter,
  ZAxis,
  ComposedChart,
  Bar
} from 'recharts';
import { 
  FiAlertCircle, 
  FiBarChart2, 
  FiCalendar, 
  FiClock, 
  FiFilter, 
  FiInfo, 
  FiMapPin, 
  FiRefreshCw,
  FiTrendingUp,
  FiMap,
  FiActivity
} from 'react-icons/fi';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import heatmap from 'heatmap.js';
import moment from 'moment';

// API service for fetching anomaly data
import { fetchAnomalies, fetchTrafficData, fetchAnomalyStats } from '../services/trafficApi';

// AI-assisted anomaly analysis
import { analyzeAnomaly } from '../services/aiService';

// Constants
const ANOMALY_SEVERITY_COLORS = {
  low: 'green.400',
  medium: 'yellow.400',
  high: 'orange.400',
  critical: 'red.400'
};

const ANOMALY_TYPE_ICONS = {
  congestion: FiBarChart2,
  incident: FiAlertCircle,
  unusual_flow: FiTrendingUp,
  anomaly: FiActivity
};

/**
 * Real-Time Traffic Anomaly Visualization Component
 * 
 * This component provides an interactive visualization of real-time
 * traffic anomalies detected by the AI system, including:
 * - Map visualization with anomaly hotspots
 * - Time series analysis of anomaly patterns
 * - Detailed anomaly information and insights
 * - Filtering and exploration tools
 */
const AnomalyVisualization = () => {
  // State for anomaly data
  const [anomalies, setAnomalies] = useState([]);
  const [filteredAnomalies, setFilteredAnomalies] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedAnomaly, setSelectedAnomaly] = useState(null);
  const [anomalyAnalysis, setAnomalyAnalysis] = useState(null);
  
  // State for traffic data
  const [trafficData, setTrafficData] = useState([]);
  
  // State for statistics
  const [stats, setStats] = useState({
    totalAnomalies: 0,
    bySeverity: {},
    byType: {},
    trend: {}
  });
  
  // State for map
  const mapRef = useRef(null);
  const mapContainerRef = useRef(null);
  const heatmapLayerRef = useRef(null);
  
  // State for filters
  const [filters, setFilters] = useState({
    timeRange: '1h',
    severity: 'all',
    type: 'all',
    location: 'all'
  });
  
  // Auto-refresh control
  const [autoRefresh, setAutoRefresh] = useState(true);
  const refreshTimerRef = useRef(null);
  
  // UI state
  const { isOpen: isAlertOpen, onClose: onAlertClose, onOpen: onAlertOpen } = useDisclosure();
  const alertColor = useColorModeValue('red.100', 'red.900');
  const cardBgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Derived state
  const locations = useMemo(() => {
    const uniqueLocations = [...new Set(anomalies.map(item => item.location_id))];
    return ['all', ...uniqueLocations];
  }, [anomalies]);
  
  const anomalyTypes = useMemo(() => {
    const uniqueTypes = [...new Set(anomalies.map(item => item.anomaly_type))];
    return ['all', ...uniqueTypes];
  }, [anomalies]);
  
  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current) return;
    
    mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN;
    
    mapRef.current = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/dark-v10',
      center: [-73.96, 40.78], // Default to NYC
      zoom: 11
    });
    
    const map = mapRef.current;
    
    map.on('load', () => {
      // Add navigation control
      map.addControl(new mapboxgl.NavigationControl(), 'top-right');
      
      // Add geolocation control
      map.addControl(new mapboxgl.GeolocateControl({
        positionOptions: {
          enableHighAccuracy: true
        },
        trackUserLocation: true
      }));
      
      // Add sources and layers
      map.addSource('anomalies', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: []
        }
      });
      
      // Add heatmap layer
      map.addLayer({
        id: 'anomalies-heat',
        type: 'heatmap',
        source: 'anomalies',
        paint: {
          'heatmap-weight': [
            'interpolate', ['linear'], ['get', 'anomaly_score'],
            0, 0,
            0.5, 0.5,
            1, 2
          ],
          'heatmap-intensity': 1,
          'heatmap-color': [
            'interpolate', ['linear'], ['heatmap-density'],
            0, 'rgba(0, 255, 255, 0)',
            0.2, 'rgba(0, 255, 255, 0.5)',
            0.4, 'rgba(0, 200, 255, 0.7)',
            0.6, 'rgba(255, 200, 0, 0.8)',
            0.8, 'rgba(255, 100, 50, 0.9)',
            1, 'rgba(255, 0, 50, 1)'
          ],
          'heatmap-radius': 30,
          'heatmap-opacity': 0.8
        }
      });
      
      // Add point layer
      map.addLayer({
        id: 'anomalies-point',
        type: 'circle',
        source: 'anomalies',
        paint: {
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            8, 4,
            15, 12
          ],
          'circle-color': [
            'match', ['get', 'anomaly_severity'],
            'low', '#48BB78',
            'medium', '#ECC94B',
            'high', '#ED8936',
            'critical', '#E53E3E',
            '#718096' // default
          ],
          'circle-opacity': 0.8,
          'circle-stroke-width': 1,
          'circle-stroke-color': '#FFFFFF'
        }
      });
      
      // Add click handler for anomaly points
      map.on('click', 'anomalies-point', (e) => {
        if (e.features.length > 0) {
          const feature = e.features[0];
          const anomalyId = feature.properties.id;
          
          // Find the full anomaly data
          const clickedAnomaly = anomalies.find(a => a.id === anomalyId);
          if (clickedAnomaly) {
            setSelectedAnomaly(clickedAnomaly);
            // Get AI analysis of the anomaly
            handleAnomalyAnalysis(clickedAnomaly);
          }
        }
      });
      
      // Change cursor on hover
      map.on('mouseenter', 'anomalies-point', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      
      map.on('mouseleave', 'anomalies-point', () => {
        map.getCanvas().style.cursor = '';
      });
    });
    
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
      }
    };
  }, []);
  
  // Update map data when anomalies change
  useEffect(() => {
    if (!mapRef.current || !filteredAnomalies.length) return;
    
    const map = mapRef.current;
    
    if (!map.loaded()) {
      map.on('load', updateMapData);
      return;
    }
    
    updateMapData();
    
    function updateMapData() {
      // Convert anomalies to GeoJSON
      const geojson = {
        type: 'FeatureCollection',
        features: filteredAnomalies.map(anomaly => ({
          type: 'Feature',
          properties: {
            id: anomaly.id,
            anomaly_score: anomaly.anomaly_score,
            anomaly_type: anomaly.anomaly_type,
            anomaly_severity: anomaly.anomaly_severity,
            timestamp: anomaly.timestamp,
            location_id: anomaly.location_id
          },
          geometry: {
            type: 'Point',
            coordinates: [anomaly.longitude, anomaly.latitude]
          }
        }))
      };
      
      // Update source data
      if (map.getSource('anomalies')) {
        map.getSource('anomalies').setData(geojson);
      }
      
      // Fit map to anomalies if we have them
      if (filteredAnomalies.length > 0 && !selectedAnomaly) {
        const bounds = new mapboxgl.LngLatBounds();
        
        filteredAnomalies.forEach(anomaly => {
          bounds.extend([anomaly.longitude, anomaly.latitude]);
        });
        
        map.fitBounds(bounds, {
          padding: 50,
          maxZoom: 15,
          duration: 1000
        });
      }
    }
  }, [filteredAnomalies, selectedAnomaly]);
  
  // Zoom to selected anomaly
  useEffect(() => {
    if (!mapRef.current || !selectedAnomaly) return;
    
    const map = mapRef.current;
    
    if (!map.loaded()) {
      map.on('load', zoomToAnomaly);
      return;
    }
    
    zoomToAnomaly();
    
    function zoomToAnomaly() {
      map.flyTo({
        center: [selectedAnomaly.longitude, selectedAnomaly.latitude],
        zoom: 15,
        duration: 1500
      });
      
      // Add a popup for the selected anomaly
      new mapboxgl.Popup({
        closeOnClick: false
      })
        .setLngLat([selectedAnomaly.longitude, selectedAnomaly.latitude])
        .setHTML(`
          <div style="font-family: sans-serif; padding: 8px;">
            <h3 style="margin: 0 0 8px; font-weight: bold;">${selectedAnomaly.anomaly_type.replace('_', ' ')}</h3>
            <p style="margin: 0 0 4px;">Severity: <span style="font-weight: bold; color: ${getSeverityColor(selectedAnomaly.anomaly_severity)}">${selectedAnomaly.anomaly_severity}</span></p>
            <p style="margin: 0 0 4px;">Score: ${(selectedAnomaly.anomaly_score * 100).toFixed(2)}%</p>
            <p style="margin: 0 0 4px;">Time: ${formatTime(selectedAnomaly.timestamp)}</p>
          </div>
        `)
        .addTo(map);
    }
    
    function getSeverityColor(severity) {
      switch(severity) {
        case 'low': return '#48BB78';
        case 'medium': return '#ECC94B';
        case 'high': return '#ED8936';
        case 'critical': return '#E53E3E';
        default: return '#718096';
      }
    }
    
    function formatTime(timestamp) {
      return moment(timestamp).format('HH:mm:ss');
    }
  }, [selectedAnomaly]);
  
  // Fetch anomaly data
  const fetchData = async () => {
    try {
      setIsLoading(true);
      
      // Get time range in milliseconds
      const timeRangeMs = getTimeRangeMs(filters.timeRange);
      
      // Fetch anomalies
      const anomalyData = await fetchAnomalies({
        startTime: Date.now() - timeRangeMs,
        endTime: Date.now(),
        severity: filters.severity === 'all' ? undefined : filters.severity,
        type: filters.type === 'all' ? undefined : filters.type,
        location: filters.location === 'all' ? undefined : filters.location
      });
      
      // Fetch traffic data
      const trafficResponse = await fetchTrafficData({
        startTime: Date.now() - timeRangeMs,
        endTime: Date.now(),
        location: filters.location === 'all' ? undefined : filters.location
      });
      
      // Fetch stats
      const statsResponse = await fetchAnomalyStats({
        startTime: Date.now() - timeRangeMs,
        endTime: Date.now()
      });
      
      setAnomalies(anomalyData);
      setTrafficData(trafficResponse);
      setStats(statsResponse);
      
      // Show alert for critical anomalies
      const criticalAnomalies = anomalyData.filter(a => a.anomaly_severity === 'critical');
      if (criticalAnomalies.length > 0) {
        onAlertOpen();
      }
      
      setError(null);
    } catch (err) {
      console.error('Error fetching anomaly data:', err);
      setError('Failed to fetch anomaly data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Apply filters to anomalies
  useEffect(() => {
    if (!anomalies.length) return;
    
    let filtered = [...anomalies];
    
    if (filters.severity !== 'all') {
      filtered = filtered.filter(a => a.anomaly_severity === filters.severity);
    }
    
    if (filters.type !== 'all') {
      filtered = filtered.filter(a => a.anomaly_type === filters.type);
    }
    
    if (filters.location !== 'all') {
      filtered = filtered.filter(a => a.location_id === filters.location);
    }
    
    setFilteredAnomalies(filtered);
  }, [anomalies, filters]);
  
  // Initial data load and auto-refresh setup
  useEffect(() => {
    fetchData();
    
    // Set up auto-refresh
    if (autoRefresh) {
      refreshTimerRef.current = setInterval(fetchData, 30000); // Refresh every 30 seconds
    }
    
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [filters, autoRefresh]);
  
  // Update auto-refresh when toggle changes
  useEffect(() => {
    if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current);
    }
    
    if (autoRefresh) {
      refreshTimerRef.current = setInterval(fetchData, 30000);
    }
  }, [autoRefresh]);
  
  // Handle filter changes
  const handleFilterChange = (filterName, value) => {
    setFilters(prev => ({
      ...prev,
      [filterName]: value
    }));
    
    // Reset selected anomaly when filters change
    setSelectedAnomaly(null);
  };
  
  // Get AI analysis for selected anomaly
  const handleAnomalyAnalysis = async (anomaly) => {
    try {
      const analysis = await analyzeAnomaly(anomaly);
      setAnomalyAnalysis(analysis);
    } catch (err) {
      console.error('Error getting anomaly analysis:', err);
      setAnomalyAnalysis({
        description: 'Unable to generate analysis at this time.',
        possibleCauses: [],
        recommendations: []
      });
    }
  };
  
  // Helper function to get time range in milliseconds
  const getTimeRangeMs = (timeRange) => {
    switch(timeRange) {
      case '15m': return 15 * 60 * 1000;
      case '30m': return 30 * 60 * 1000;
      case '1h': return 60 * 60 * 1000;
      case '3h': return 3 * 60 * 60 * 1000;
      case '6h': return 6 * 60 * 60 * 1000;
      case '12h': return 12 * 60 * 60 * 1000;
      case '24h': return 24 * 60 * 60 * 1000;
      default: return 60 * 60 * 1000;
    }
  };
  
  // Helper function to format timestamps
  const formatTimestamp = (timestamp) => {
    return moment(timestamp).format('HH:mm:ss');
  };
  
  // Render anomaly severity badge
  const renderSeverityBadge = (severity) => {
    return (
      <Badge 
        colorScheme={
          severity === 'critical' ? 'red' :
          severity === 'high' ? 'orange' :
          severity === 'medium' ? 'yellow' : 'green'
        }
        fontSize="0.8em"
        px={2}
        py={1}
        borderRadius="full"
      >
        {severity}
      </Badge>
    );
  };
  
  // Render anomaly type with icon
  const renderAnomalyType = (type) => {
    const IconComponent = ANOMALY_TYPE_ICONS[type] || FiInfo;
    return (
      <Flex align="center">
        <Icon as={IconComponent} mr={2} />
        <Text>{type.replace('_', ' ')}</Text>
      </Flex>
    );
  };
  
  return (
    <Box>
      <Heading mb={4} size="lg">Real-Time Traffic Anomaly Detection</Heading>
      
      {/* Alert for critical anomalies */}
      {isAlertOpen && (
        <Alert 
          status="error" 
          mb={4}
          borderRadius="md"
          bg={alertColor}
        >
          <AlertIcon />
          <Box flex="1">
            <AlertTitle>Critical Anomalies Detected!</AlertTitle>
            <AlertDescription display="block">
              {filteredAnomalies.filter(a => a.anomaly_severity === 'critical').length} critical anomalies 
              detected in the current time frame. Immediate attention required.
            </AlertDescription>
          </Box>
          <CloseButton 
            position="absolute" 
            right="8px" 
            top="8px" 
            onClick={onAlertClose} 
          />
        </Alert>
      )}
      
      {/* Filter controls */}
      <Flex 
        mb={4} 
        p={4} 
        borderWidth="1px" 
        borderRadius="lg" 
        bg={cardBgColor}
        borderColor={borderColor}
        direction={{ base: 'column', md: 'row' }}
        align={{ base: 'stretch', md: 'center' }}
        wrap="wrap"
        gap={4}
      >
        <Box flex="1" minW="150px">
          <FormControl>
            <FormLabel fontSize="sm" mb={1}>Time Range</FormLabel>
            <Select 
              size="sm"
              value={filters.timeRange}
              onChange={(e) => handleFilterChange('timeRange', e.target.value)}
              icon={<FiClock />}
            >
              <option value="15m">Last 15 minutes</option>
              <option value="30m">Last 30 minutes</option>
              <option value="1h">Last hour</option>
              <option value="3h">Last 3 hours</option>
              <option value="6h">Last 6 hours</option>
              <option value="12h">Last 12 hours</option>
              <option value="24h">Last 24 hours</option>
            </Select>
          </FormControl>
        </Box>
        
        <Box flex="1" minW="150px">
          <FormControl>
            <FormLabel fontSize="sm" mb={1}>Severity</FormLabel>
            <Select 
              size="sm"
              value={filters.severity}
              onChange={(e) => handleFilterChange('severity', e.target.value)}
              icon={<FiFilter />}
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </Select>
          </FormControl>
        </Box>
        
        <Box flex="1" minW="150px">
          <FormControl>
            <FormLabel fontSize="sm" mb={1}>Anomaly Type</FormLabel>
            <Select
              size="sm"
              value={filters.type}
              onChange={(e) => handleFilterChange('type', e.target.value)}
              icon={<FiFilter />}
            >
              <option value="all">All Types</option>
              {anomalyTypes.filter(t => t !== 'all').map(type => (
                <option key={type} value={type}>{type.replace('_', ' ')}</option>
              ))}
            </Select>
          </FormControl>
        </Box>
        
        <Box flex="1" minW="150px">
          <FormControl>
            <FormLabel fontSize="sm" mb={1}>Location</FormLabel>
            <Select
              size="sm"
              value={filters.location}
              onChange={(e) => handleFilterChange('location', e.target.value)}
              icon={<FiMapPin />}
            >
              <option value="all">All Locations</option>
              {locations.filter(l => l !== 'all').map(locationId => (
                <option key={locationId} value={locationId}>Location {locationId}</option>
              ))}
            </Select>
          </FormControl>
        </Box>
        
        <Flex align="flex-end" mt={{ base: 2, md: 0 }}>
          <Button
            size="sm"
            leftIcon={<FiRefreshCw />}
            onClick={fetchData}
            mr={2}
            isLoading={isLoading}
          >
            Refresh
          </Button>
          
          <Button
            size="sm"
            colorScheme={autoRefresh ? "green" : "gray"}
            variant="outline"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? "Auto-refresh: On" : "Auto-refresh: Off"}
          </Button>
        </Flex>
      </Flex>
      
      {/* Main content */}
      <Grid
        templateColumns={{ base: '1fr', lg: '3fr 1fr' }}
        gap={4}
      >
        {/* Left column - Map and charts */}
        <GridItem>
          {/* Map */}
          <Box 
            mb={4} 
            borderRadius="lg" 
            overflow="hidden" 
            borderWidth="1px"
            borderColor={borderColor}
            position="relative"
            height="500px"
          >
            <Box 
              ref={mapContainerRef} 
              position="absolute" 
              top={0} 
              left={0} 
              right={0} 
              bottom={0} 
            />
            
            {isLoading && (
              <Flex 
                position="absolute" 
                top={0} 
                left={0} 
                right={0} 
                bottom={0} 
                bg="rgba(0,0,0,0.2)" 
                zIndex={10}
                justify="center"
                align="center"
              >
                <Spinner size="xl" color="blue.500" />
              </Flex>
            )}
          </Box>
          
          {/* Traffic metrics charts */}
          <Box
            mb={4}
            p={4}
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBgColor}
            borderColor={borderColor}
          >
            <Heading size="md" mb={4}>Traffic Metrics with Anomaly Detection</Heading>
            
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart
                data={trafficData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={formatTimestamp}
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  yAxisId="left"
                  tick={{ fontSize: 12 }}
                  label={{ 
                    value: 'Traffic Volume', 
                    angle: -90, 
                    position: 'insideLeft',
                    style: { textAnchor: 'middle' }
                  }}
                />
                <YAxis 
                  yAxisId="right" 
                  orientation="right"
                  domain={[0, 100]}
                  tick={{ fontSize: 12 }}
                  label={{ 
                    value: 'Speed (km/h)', 
                    angle: 90, 
                    position: 'insideRight',
                    style: { textAnchor: 'middle' }
                  }}
                />
                <RechartsTooltip
                  formatter={(value, name) => [value, name]}
                  labelFormatter={formatTimestamp}
                  contentStyle={{ 
                    backgroundColor: cardBgColor, 
                    borderColor: borderColor 
                  }}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="volume"
                  yAxisId="left"
                  fill="#5A67D8"
                  stroke="#5A67D8"
                  fillOpacity={0.3}
                  name="Traffic Volume"
                />
                <Line
                  type="monotone"
                  dataKey="speed"
                  yAxisId="right"
                  stroke="#38B2AC"
                  name="Speed"
                  strokeWidth={2}
                />
                <Bar
                  dataKey="anomaly_score"
                  yAxisId="left"
                  barSize={20}
                  fill="#F56565"
                  name="Anomaly Score"
                  fillOpacity={0.7}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </Box>
          
          {/* Anomaly distribution chart */}
          <Box
            mb={4}
            p={4}
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBgColor}
            borderColor={borderColor}
          >
            <Heading size="md" mb={4}>Anomaly Score Distribution</Heading>
            
            <ResponsiveContainer width="100%" height={250}>
              <ScatterChart
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis 
                  dataKey="timestamp" 
                  name="Time"
                  tickFormatter={formatTimestamp}
                  type="category"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  dataKey="anomaly_score" 
                  name="Anomaly Score" 
                  domain={[0, 1]}
                  tick={{ fontSize: 12 }}
                  label={{ 
                    value: 'Anomaly Score', 
                    angle: -90, 
                    position: 'insideLeft',
                    style: { textAnchor: 'middle' }
                  }}
                />
                <ZAxis 
                  dataKey="measured_values.volume" 
                  range={[50, 400]} 
                  name="Volume" 
                />
                <RechartsTooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  formatter={(value, name) => {
                    if (name === 'Anomaly Score') {
                      return [(value * 100).toFixed(2) + '%', name];
                    }
                    return [value, name];
                  }}
                  labelFormatter={formatTimestamp}
                  contentStyle={{ 
                    backgroundColor: cardBgColor, 
                    borderColor: borderColor 
                  }}
                />
                <Legend />
                <Scatter 
                  name="Anomalies" 
                  data={filteredAnomalies} 
                  fill="#F56565"
                  line={{ stroke: '#F56565', strokeWidth: 1, strokeDasharray: '5 5' }}
                  shape={(props) => {
                    const { cx, cy, fill, payload } = props;
                    
                    // Determine color based on severity
                    let color;
                    switch(payload.anomaly_severity) {
                      case 'critical': color = '#E53E3E'; break;
                      case 'high': color = '#DD6B20'; break;
                      case 'medium': color = '#D69E2E'; break;
                      case 'low': color = '#38A169'; break;
                      default: color = '#718096';
                    }
                    
                    return (
                      <svg>
                        <circle
                          cx={cx}
                          cy={cy}
                          r={8}
                          fill={color}
                          stroke="#FFFFFF"
                          strokeWidth={1}
                          opacity={0.8}
                        />
                      </svg>
                    );
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </Box>
        </GridItem>
        
        {/* Right column - Stats and details */}
        <GridItem>
          {/* Anomaly statistics */}
          <Box
            mb={4}
            p={4}
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBgColor}
            borderColor={borderColor}
          >
            <Heading size="md" mb={4}>Anomaly Statistics</Heading>
            
            <Grid templateColumns="1fr 1fr" gap={4}>
              <Stat>
                <StatLabel>Total Anomalies</StatLabel>
                <StatNumber>{stats.totalAnomalies}</StatNumber>
                <StatHelpText>
                  In selected time range
                </StatHelpText>
              </Stat>
              
              <Stat>
                <StatLabel>Critical Anomalies</StatLabel>
                <StatNumber>{stats.bySeverity?.critical || 0}</StatNumber>
                <StatHelpText>
                  <Text color="red.500" fontWeight="bold">
                    Requires immediate attention
                  </Text>
                </StatHelpText>
              </Stat>
            </Grid>
            
            <Heading size="sm" mt={6} mb={3}>Distribution by Type</Heading>
            <Grid templateColumns="1fr 1fr" gap={4} mb={4}>
              {Object.entries(stats.byType || {}).map(([type, count]) => (
                <Stat key={type} size="sm">
                  <StatLabel>{type.replace('_', ' ')}</StatLabel>
                  <StatNumber>{count}</StatNumber>
                </Stat>
              ))}
            </Grid>
            
            <Heading size="sm" mt={6} mb={3}>Distribution by Severity</Heading>
            <Grid templateColumns="1fr 1fr" gap={4} mb={2}>
              {Object.entries(stats.bySeverity || {}).map(([severity, count]) => (
                <Stat key={severity} size="sm">
                  <StatLabel>
                    <Flex align="center">
                      {renderSeverityBadge(severity)}
                    </Flex>
                  </StatLabel>
                  <StatNumber>{count}</StatNumber>
                </Stat>
              ))}
            </Grid>
          </Box>
          
          {/* Selected anomaly details */}
          {selectedAnomaly && (
            <Box
              mb={4}
              p={4}
              borderWidth="1px"
              borderRadius="lg"
              bg={cardBgColor}
              borderColor={borderColor}
            >
              <Flex justify="space-between" align="center" mb={3}>
                <Heading size="md">Anomaly Details</Heading>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSelectedAnomaly(null)}
                >
                  Close
                </Button>
              </Flex>
              
              <Box mb={4}>
                <Flex justify="space-between" mb={2}>
                  <Text fontWeight="bold">Type:</Text>
                  <Box>{renderAnomalyType(selectedAnomaly.anomaly_type)}</Box>
                </Flex>
                
                <Flex justify="space-between" mb={2}>
                  <Text fontWeight="bold">Severity:</Text>
                  <Box>{renderSeverityBadge(selectedAnomaly.anomaly_severity)}</Box>
                </Flex>
                
                <Flex justify="space-between" mb={2}>
                  <Text fontWeight="bold">Score:</Text>
                  <Text>{(selectedAnomaly.anomaly_score * 100).toFixed(2)}%</Text>
                </Flex>
                
                <Flex justify="space-between" mb={2}>
                  <Text fontWeight="bold">Time:</Text>
                  <Text>{moment(selectedAnomaly.timestamp).format('YYYY-MM-DD HH:mm:ss')}</Text>
                </Flex>
                
                <Flex justify="space-between" mb={2}>
                  <Text fontWeight="bold">Location:</Text>
                  <Text>ID: {selectedAnomaly.location_id}</Text>
                </Flex>
              </Box>
              
              <Heading size="sm" mb={3}>Measured Values</Heading>
              <Grid templateColumns="1fr 1fr" gap={2} mb={4}>
                {Object.entries(selectedAnomaly.measured_values || {}).map(([key, value]) => (
                  <Flex key={key} justify="space-between" px={2} py={1} borderWidth="1px" borderRadius="md">
                    <Text fontSize="sm" fontWeight="bold">{key}:</Text>
                    <Text fontSize="sm">{value.toFixed(2)}</Text>
                  </Flex>
                ))}
              </Grid>
              
              {/* AI-assisted analysis */}
              {anomalyAnalysis ? (
                <Box mt={4} p={3} borderWidth="1px" borderRadius="md" borderColor="blue.200" bg="blue.50">
                  <Heading size="sm" mb={2} color="blue.600">
                    <Flex align="center">
                      <Icon as={FiInfo} mr={2} />
                      AI-Assisted Analysis
                    </Flex>
                  </Heading>
                  
                  <Text fontSize="sm" mb={3}>{anomalyAnalysis.description}</Text>
                  
                  {anomalyAnalysis.possibleCauses && anomalyAnalysis.possibleCauses.length > 0 && (
                    <Box mb={3}>
                      <Text fontSize="sm" fontWeight="bold">Possible Causes:</Text>
                      <ul style={{ paddingLeft: '1.5rem', marginTop: '0.25rem' }}>
                        {anomalyAnalysis.possibleCauses.map((cause, i) => (
                          <li key={i}>
                            <Text fontSize="sm">{cause}</Text>
                          </li>
                        ))}
                      </ul>
                    </Box>
                  )}
                  
                  {anomalyAnalysis.recommendations && anomalyAnalysis.recommendations.length > 0 && (
                    <Box>
                      <Text fontSize="sm" fontWeight="bold">Recommendations:</Text>
                      <ul style={{ paddingLeft: '1.5rem', marginTop: '0.25rem' }}>
                        {anomalyAnalysis.recommendations.map((rec, i) => (
                          <li key={i}>
                            <Text fontSize="sm">{rec}</Text>
                          </li>
                        ))}
                      </ul>
                    </Box>
                  )}
                </Box>
              ) : (
                <Flex justify="center" mt={4}>
                  <Spinner size="sm" mr={2} />
                  <Text>Analyzing anomaly patterns...</Text>
                </Flex>
              )}
            </Box>
          )}
          
          {/* Recent anomalies list */}
          <Box
            mb={4}
            p={4}
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBgColor}
            borderColor={borderColor}
            maxH="500px"
            overflowY="auto"
          >
            <Heading size="md" mb={4}>Recent Anomalies</Heading>
            
            {filteredAnomalies.length === 0 ? (
              <Flex 
                direction="column" 
                align="center" 
                justify="center" 
                py={8}
                color="gray.500"
              >
                <Icon as={FiInfo} fontSize="3xl" mb={3} />
                <Text>No anomalies match the current filters</Text>
              </Flex>
            ) : (
              filteredAnomalies
                .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                .slice(0, 20)
                .map(anomaly => (
                  <Box 
                    key={anomaly.id}
                    mb={3}
                    p={3}
                    borderWidth="1px"
                    borderRadius="md"
                    borderColor={borderColor}
                    _hover={{ bg: 'gray.50', cursor: 'pointer' }}
                    onClick={() => {
                      setSelectedAnomaly(anomaly);
                      handleAnomalyAnalysis(anomaly);
                    }}
                  >
                    <Flex justify="space-between" align="center" mb={2}>
                      <Flex align="center">
                        <Icon 
                          as={ANOMALY_TYPE_ICONS[anomaly.anomaly_type] || FiInfo} 
                          mr={2}
                          color={
                            anomaly.anomaly_severity === 'critical' ? 'red.500' :
                            anomaly.anomaly_severity === 'high' ? 'orange.500' :
                            anomaly.anomaly_severity === 'medium' ? 'yellow.500' : 'green.500'
                          }
                        />
                        <Text fontWeight="bold">{anomaly.anomaly_type.replace('_', ' ')}</Text>
                      </Flex>
                      {renderSeverityBadge(anomaly.anomaly_severity)}
                    </Flex>
                    
                    <Flex justify="space-between" fontSize="sm">
                      <Flex align="center">
                        <Icon as={FiMapPin} fontSize="xs" mr={1} />
                        <Text>Location {anomaly.location_id}</Text>
                      </Flex>
                      <Flex align="center">
                        <Icon as={FiClock} fontSize="xs" mr={1} />
                        <Text>{moment(anomaly.timestamp).fromNow()}</Text>
                      </Flex>
                    </Flex>
                    
                    <Flex justify="space-between" mt={2} fontSize="sm">
                      <Text color="gray.600">Score: {(anomaly.anomaly_score * 100).toFixed(2)}%</Text>
                      <Text 
                        color="blue.500"
                        fontWeight="medium"
                      >
                        View details
                      </Text>
                    </Flex>
                  </Box>
                ))
            )}
          </Box>
        </GridItem>
      </Grid>
    </Box>
  );
};

export default AnomalyVisualization;