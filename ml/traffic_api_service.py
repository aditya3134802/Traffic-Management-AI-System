"""
Traffic API Service

This module provides APIs for traffic data and anomaly detection.
It simulates real-time traffic data and anomaly detection results.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import uuid
import json
import logging
import time
import random
from datetime import datetime, timedelta
import asyncio
import os

# Import anomaly detection service
from anomaly_detection_service import TrafficAnomalyDetector, AnomalyDetectionConfig

# Initialize the FastAPI app
app = FastAPI(
    title="Traffic Management API",
    description="API for real-time traffic data and anomaly detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("traffic-api")

# Initialize the anomaly detector
config = AnomalyDetectionConfig()
anomaly_detector = TrafficAnomalyDetector(config)

# In-memory data store (for demo purposes)
traffic_data = []
anomalies = []
locations = []


# Models
class Location(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    type: str  # intersection, highway, urban, etc.
    description: Optional[str] = None
    lane_count: Optional[int] = None


class TrafficDataPoint(BaseModel):
    id: str
    timestamp: str
    location_id: str
    speed: float  # Average speed in km/h
    volume: int  # Number of vehicles
    occupancy: float  # Percentage of time sensor is occupied
    density: Optional[float] = None  # Vehicles per km
    speed_std: Optional[float] = None  # Standard deviation of speed
    expected_volume: Optional[float] = None  # Expected volume based on historical data
    travel_time: Optional[float] = None  # Travel time in seconds
    congestion_level: Optional[float] = None  # 0-1 congestion level
    measured_values: Optional[Dict[str, float]] = None  # Additional metrics


class Anomaly(BaseModel):
    id: str
    timestamp: str
    location_id: str
    anomaly_score: float
    anomaly_type: str  # congestion, incident, unusual_flow, anomaly
    anomaly_severity: str  # low, medium, high, critical
    measured_values: Dict[str, float]
    latitude: float
    longitude: float
    detection_time: str


class AnomalyStats(BaseModel):
    totalAnomalies: int
    bySeverity: Dict[str, int]
    byType: Dict[str, int]
    trend: Dict[str, Any]


class AIAnalysis(BaseModel):
    description: str
    possibleCauses: List[str]
    recommendations: List[str]
    confidence: float
    relatedIncidents: Optional[List[Dict[str, Any]]] = None


# Helper function to generate random traffic data
def generate_traffic_data(num_locations=10, data_points_per_location=100):
    global traffic_data, anomalies, locations
    
    # Generate locations if empty
    if not locations:
        for i in range(num_locations):
            # NYC area coordinates (for demo)
            base_lat = 40.7128
            base_lng = -74.0060
            
            # Location type distribution
            loc_types = ["intersection", "highway", "urban", "suburban", "arterial"]
            loc_type = random.choice(loc_types)
            
            # Add some geographical spread
            lat = base_lat + (random.random() - 0.5) * 0.1
            lng = base_lng + (random.random() - 0.5) * 0.1
            
            locations.append({
                "id": f"L{i+1:03d}",
                "name": f"Location {i+1}",
                "latitude": lat,
                "longitude": lng,
                "type": loc_type,
                "description": f"Traffic monitoring point at {loc_type}",
                "lane_count": random.randint(2, 8)
            })
    
    # Generate traffic data
    traffic_data = []
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    time_interval = timedelta(minutes=5)
    current_time = start_time
    
    # Base patterns for different location types
    patterns = {
        "intersection": {"base_speed": 30, "base_volume": 200, "variance_speed": 10, "variance_volume": 50},
        "highway": {"base_speed": 90, "base_volume": 500, "variance_speed": 20, "variance_volume": 100},
        "urban": {"base_speed": 40, "base_volume": 300, "variance_speed": 15, "variance_volume": 80},
        "suburban": {"base_speed": 60, "base_volume": 150, "variance_speed": 10, "variance_volume": 40},
        "arterial": {"base_speed": 50, "base_volume": 250, "variance_speed": 12, "variance_volume": 60},
    }
    
    # Time-based patterns (rush hours, etc.)
    def get_time_factor(dt):
        hour = dt.hour
        minute = dt.minute
        
        # Morning rush hour: 7-9 AM
        if 7 <= hour < 9:
            return 1.5
        # Evening rush hour: 4-7 PM
        elif 16 <= hour < 19:
            return 1.4
        # Late night: 11 PM - 5 AM
        elif hour >= 23 or hour < 5:
            return 0.4
        # Default: normal traffic
        else:
            return 1.0
    
    # Generate data for each location and timepoint
    while current_time <= end_time:
        for location in locations:
            location_id = location["id"]
            location_type = location["type"]
            pattern = patterns[location_type]
            
            # Apply time-based patterns
            time_factor = get_time_factor(current_time)
            
            # Add some randomness with time consistency
            seed = int(current_time.timestamp()) + int(location_id[1:])
            r = random.Random(seed)
            
            # Calculate base metrics with random variation
            speed = max(5, min(130, pattern["base_speed"] * time_factor * (0.8 + 0.4 * r.random())))
            volume = max(10, min(1000, pattern["base_volume"] * time_factor * (0.8 + 0.4 * r.random())))
            
            # Inverse relationship between volume and speed at high volumes
            if volume > pattern["base_volume"] * 1.2:
                speed = speed * (1.0 - (volume / (pattern["base_volume"] * 2.5)))
            
            # Derived metrics
            occupancy = min(100, volume / 10)
            density = volume / max(1, speed)
            speed_std = pattern["variance_speed"] * (0.5 + 0.8 * r.random())
            
            # Expected volume based on historical patterns
            expected_volume = pattern["base_volume"] * time_factor
            
            # Congestion level
            congestion_level = min(1.0, max(0.0, 1.0 - (speed / pattern["base_speed"])))
            
            # Create data point
            data_point = {
                "id": str(uuid.uuid4()),
                "timestamp": current_time.isoformat(),
                "location_id": location_id,
                "speed": speed,
                "volume": int(volume),
                "occupancy": occupancy,
                "density": density,
                "speed_std": speed_std,
                "expected_volume": expected_volume,
                "travel_time": 1000 / max(1, speed),
                "congestion_level": congestion_level,
                "measured_values": {
                    "speed": speed,
                    "volume": volume,
                    "occupancy": occupancy,
                    "density": density,
                    "speed_std": speed_std,
                    "expected_volume": expected_volume,
                    "travel_time": 1000 / max(1, speed),
                    "congestion_level": congestion_level
                }
            }
            
            traffic_data.append(data_point)
        
        current_time += time_interval
    
    # Sort data by timestamp
    traffic_data.sort(key=lambda x: x["timestamp"])
    
    # Generate anomalies
    generate_anomalies()


def generate_anomalies():
    global traffic_data, anomalies, locations
    
    if not traffic_data:
        return
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(traffic_data)
    
    # Initialize anomaly detector if not trained
    if not hasattr(anomaly_detector, 'is_trained') or not anomaly_detector.is_trained:
        # Train with a subset of the data
        train_df = df.sample(frac=0.7)
        anomaly_detector.train(train_df)
    
    # Detect anomalies in the full dataset
    result_df = anomaly_detector.detect_anomalies(df)
    
    # Filter to only include anomalies
    anomaly_df = result_df[result_df["is_anomaly"]]
    
    # Convert back to list of dicts
    anomalies = []
    for _, row in anomaly_df.iterrows():
        # Find location details
        location = next((loc for loc in locations if loc["id"] == row["location_id"]), None)
        
        if not location:
            continue
        
        # Convert row to dict, keeping only needed columns
        anomaly = {
            "id": str(uuid.uuid4()),
            "timestamp": row["timestamp"],
            "location_id": row["location_id"],
            "anomaly_score": float(row["anomaly_score"]),
            "anomaly_type": row["anomaly_type"],
            "anomaly_severity": row["anomaly_severity"],
            "measured_values": {
                k: float(row[k]) for k in [
                    "speed", "volume", "occupancy", "density", 
                    "speed_std", "expected_volume", "travel_time", 
                    "congestion_level"
                ] if k in row and not pd.isna(row[k])
            },
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "detection_time": datetime.now().isoformat()
        }
        
        anomalies.append(anomaly)


def get_ai_analysis(anomaly):
    """Generate simulated AI analysis for an anomaly"""
    
    anomaly_type = anomaly.get("anomaly_type", "anomaly")
    severity = anomaly.get("anomaly_severity", "medium")
    
    # Basic descriptions based on anomaly type
    descriptions = {
        "congestion": "This anomaly indicates unusual traffic congestion not following typical patterns.",
        "incident": "This appears to be a traffic incident causing disruption to normal flow.",
        "unusual_flow": "An atypical traffic flow pattern has been detected in this area.",
        "anomaly": "Unusual traffic behavior detected that doesn't match historical patterns."
    }
    
    # Possible causes based on type
    causes = {
        "congestion": [
            "Higher than expected traffic volume for this time period",
            "Reduced road capacity due to construction or lane closure",
            "Spillover congestion from nearby road segments",
            "Special event increasing traffic demand",
            "Weather conditions affecting driver behavior"
        ],
        "incident": [
            "Vehicle collision or breakdown",
            "Road obstruction or debris",
            "Emergency vehicle activity",
            "Traffic signal malfunction",
            "Road surface issue (pothole, ice, flooding)"
        ],
        "unusual_flow": [
            "Rerouting due to congestion on alternative routes",
            "Navigation app recommendations causing unexpected patterns",
            "Temporary road closure or detour in effect",
            "Unexpected event generating traffic",
            "Unusual commuting pattern due to holiday or special event"
        ],
        "anomaly": [
            "Sensor malfunction or calibration issue",
            "Unexpected traffic behavior due to external factors",
            "Combination of multiple minor factors",
            "Emerging traffic pattern not yet incorporated in baseline",
            "Data processing or transmission issue"
        ]
    }
    
    # Recommendations based on type and severity
    recommendations = {
        "congestion": {
            "low": [
                "Monitor situation for further development",
                "Prepare for potential traffic management if conditions worsen"
            ],
            "medium": [
                "Adjust signal timing to improve flow",
                "Send traffic advisory to navigation providers",
                "Consider activating variable message signs if available"
            ],
            "high": [
                "Implement pre-planned congestion management protocol",
                "Adjust signal timing across multiple intersections",
                "Deploy traffic management personnel if available"
            ],
            "critical": [
                "Activate emergency traffic management plan",
                "Implement all available congestion mitigation measures",
                "Consider alternative routing strategies",
                "Coordinate with emergency services for critical situation management"
            ]
        },
        "incident": {
            "low": [
                "Dispatch operator to verify incident via camera",
                "Monitor for resolution without intervention"
            ],
            "medium": [
                "Dispatch incident response unit to assess",
                "Adjust signal timing to accommodate disruption",
                "Issue traffic advisory for affected area"
            ],
            "high": [
                "Dispatch emergency services and incident management team",
                "Implement incident management protocol",
                "Issue traffic warnings and suggest alternative routes"
            ],
            "critical": [
                "Activate emergency response protocol",
                "Coordinate with police, fire, and medical services",
                "Implement emergency traffic management plan",
                "Issue widespread alerts and detour information"
            ]
        },
        "unusual_flow": {
            "low": [
                "Monitor for pattern development",
                "Log for future pattern analysis"
            ],
            "medium": [
                "Check for potential causes (events, construction, etc.)",
                "Adjust traffic management strategy if pattern persists"
            ],
            "high": [
                "Implement adaptive traffic management response",
                "Investigate root cause and address if possible",
                "Coordinate with other traffic management systems"
            ],
            "critical": [
                "Dispatch traffic management personnel to affected areas",
                "Implement emergency traffic flow adjustments",
                "Coordinate with regional traffic management centers",
                "Issue public advisories about unusual conditions"
            ]
        },
        "anomaly": {
            "low": [
                "Flag for data quality review",
                "Monitor for pattern emergence"
            ],
            "medium": [
                "Verify sensor functionality",
                "Check for correlated anomalies in nearby locations",
                "Review recent maintenance or configuration changes"
            ],
            "high": [
                "Dispatch technician to inspect sensor equipment",
                "Implement backup data collection if available",
                "Review system integrity across monitoring network"
            ],
            "critical": [
                "Initiate system-wide diagnostic tests",
                "Switch to backup systems if available",
                "Dispatch emergency maintenance team",
                "Implement manual traffic monitoring protocols"
            ]
        }
    }
    
    # Select a few random causes and recommendations based on type and severity
    selected_causes = random.sample(causes.get(anomaly_type, causes["anomaly"]), min(3, len(causes.get(anomaly_type, causes["anomaly"]))))
    
    type_recommendations = recommendations.get(anomaly_type, recommendations["anomaly"])
    severity_recommendations = type_recommendations.get(severity, type_recommendations["medium"])
    selected_recommendations = random.sample(severity_recommendations, min(3, len(severity_recommendations)))
    
    # Calculate confidence based on severity and anomaly score
    severity_factor = {
        "low": 0.7,
        "medium": 0.8,
        "high": 0.9,
        "critical": 0.95
    }
    
    confidence = severity_factor.get(severity, 0.8) * anomaly.get("anomaly_score", 0.5)
    
    # Generate final analysis
    analysis = {
        "description": descriptions.get(anomaly_type, descriptions["anomaly"]),
        "possibleCauses": selected_causes,
        "recommendations": selected_recommendations,
        "confidence": confidence
    }
    
    return analysis


# Initialize data
generate_traffic_data()


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Traffic Management API is running"}


@app.get("/api/traffic-data", response_model=List[TrafficDataPoint])
async def get_traffic_data(
    startTime: Optional[int] = Query(None, description="Start timestamp in milliseconds"),
    endTime: Optional[int] = Query(None, description="End timestamp in milliseconds"),
    location: Optional[str] = Query(None, description="Filter by location ID")
):
    global traffic_data
    
    # Filter by time range if provided
    filtered_data = traffic_data
    
    if startTime:
        start_dt = datetime.fromtimestamp(startTime / 1000)
        filtered_data = [d for d in filtered_data if datetime.fromisoformat(d["timestamp"].replace('Z', '+00:00')) >= start_dt]
    
    if endTime:
        end_dt = datetime.fromtimestamp(endTime / 1000)
        filtered_data = [d for d in filtered_data if datetime.fromisoformat(d["timestamp"].replace('Z', '+00:00')) <= end_dt]
    
    # Filter by location if provided
    if location:
        filtered_data = [d for d in filtered_data if d["location_id"] == location]
    
    # Return the latest 1000 data points for performance
    return filtered_data[-1000:]


@app.get("/api/anomalies", response_model=List[Anomaly])
async def get_anomalies(
    startTime: Optional[int] = Query(None, description="Start timestamp in milliseconds"),
    endTime: Optional[int] = Query(None, description="End timestamp in milliseconds"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    type: Optional[str] = Query(None, description="Filter by anomaly type"),
    location: Optional[str] = Query(None, description="Filter by location ID")
):
    global anomalies
    
    # Filter by time range if provided
    filtered_anomalies = anomalies
    
    if startTime:
        start_dt = datetime.fromtimestamp(startTime / 1000)
        filtered_anomalies = [a for a in filtered_anomalies if datetime.fromisoformat(a["timestamp"].replace('Z', '+00:00')) >= start_dt]
    
    if endTime:
        end_dt = datetime.fromtimestamp(endTime / 1000)
        filtered_anomalies = [a for a in filtered_anomalies if datetime.fromisoformat(a["timestamp"].replace('Z', '+00:00')) <= end_dt]
    
    # Apply additional filters
    if severity:
        filtered_anomalies = [a for a in filtered_anomalies if a["anomaly_severity"] == severity]
    
    if type:
        filtered_anomalies = [a for a in filtered_anomalies if a["anomaly_type"] == type]
    
    if location:
        filtered_anomalies = [a for a in filtered_anomalies if a["location_id"] == location]
    
    return filtered_anomalies


@app.get("/api/locations", response_model=List[Location])
async def get_locations():
    return locations


@app.get("/api/anomaly/stats", response_model=AnomalyStats)
async def get_anomaly_stats(
    startTime: Optional[int] = Query(None, description="Start timestamp in milliseconds"),
    endTime: Optional[int] = Query(None, description="End timestamp in milliseconds")
):
    global anomalies
    
    # Filter by time range if provided
    filtered_anomalies = anomalies
    
    if startTime:
        start_dt = datetime.fromtimestamp(startTime / 1000)
        filtered_anomalies = [a for a in filtered_anomalies if datetime.fromisoformat(a["timestamp"].replace('Z', '+00:00')) >= start_dt]
    
    if endTime:
        end_dt = datetime.fromtimestamp(endTime / 1000)
        filtered_anomalies = [a for a in filtered_anomalies if datetime.fromisoformat(a["timestamp"].replace('Z', '+00:00')) <= end_dt]
    
    # Calculate statistics
    total_anomalies = len(filtered_anomalies)
    
    # Count by severity
    severity_counts = {}
    for severity in ["low", "medium", "high", "critical"]:
        severity_counts[severity] = len([a for a in filtered_anomalies if a["anomaly_severity"] == severity])
    
    # Count by type
    type_counts = {}
    for a in filtered_anomalies:
        a_type = a["anomaly_type"]
        if a_type in type_counts:
            type_counts[a_type] += 1
        else:
            type_counts[a_type] = 1
    
    # Calculate trend (simplified)
    trend = {
        "increasing": random.choice([True, False]),
        "percentageChange": random.uniform(-15, 30),
        "timeComparisonHours": 24
    }
    
    return {
        "totalAnomalies": total_anomalies,
        "bySeverity": severity_counts,
        "byType": type_counts,
        "trend": trend
    }


@app.get("/api/anomaly/analyze/{anomaly_id}", response_model=AIAnalysis)
async def analyze_anomaly(anomaly_id: str):
    global anomalies
    
    # Find the anomaly
    anomaly = next((a for a in anomalies if a["id"] == anomaly_id), None)
    
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    
    # Generate analysis (simulating AI processing time)
    await asyncio.sleep(1.5)
    
    analysis = get_ai_analysis(anomaly)
    
    return analysis


@app.post("/api/simulate/refresh")
async def refresh_simulation():
    """Regenerate simulation data"""
    generate_traffic_data()
    return {"status": "success", "message": "Simulation data refreshed"}


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", "8000"))
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)