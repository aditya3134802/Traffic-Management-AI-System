"""
AI Analysis Service for Traffic Anomaly Detection

This module provides AI-powered analysis of traffic anomalies,
including cause identification, impact assessment, and recommendations.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import asyncio
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai-analysis-service")


class AIAnalysisService:
    """
    AI-powered analysis of traffic anomalies
    
    This service uses multiple models to:
    1. Identify the probable causes of anomalies
    2. Assess their impact on traffic patterns
    3. Generate actionable recommendations
    4. Provide natural language explanations
    """
    
    # Analysis types
    ANALYSIS_TYPES = ["basic", "detailed", "comprehensive"]
    
    # Anomaly types
    ANOMALY_TYPES = ["congestion", "incident", "unusual_flow", "anomaly"]
    
    # Severity levels
    SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
    
    def __init__(self):
        """Initialize the AI analysis service"""
        # Configuration
        self.model_path = os.environ.get("MODEL_PATH", "/models/ai-analysis")
        self.data_path = os.environ.get("DATA_PATH", "/data/traffic")
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.use_llm = os.environ.get("USE_LLM", "false").lower() == "true"
        self.llm_endpoint = os.environ.get("LLM_ENDPOINT", "http://llm-service:8080/api/generate")
        
        # Set logging level
        if hasattr(logging, self.log_level):
            logger.setLevel(getattr(logging, self.log_level))
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Load cause-recommendation database
        self._load_knowledge_base()
    
    def _initialize_models(self):
        """Initialize AI models for anomaly analysis"""
        try:
            # Cause classification model
            cause_model_path = os.path.join(self.model_path, "cause_classifier.h5")
            if os.path.exists(cause_model_path):
                logger.info(f"Loading cause classification model from {cause_model_path}")
                self.models["cause_classifier"] = load_model(cause_model_path)
            else:
                logger.warning(f"Cause classification model not found at {cause_model_path}")
                self.models["cause_classifier"] = None
            
            # Impact assessment model
            impact_model_path = os.path.join(self.model_path, "impact_assessor.h5")
            if os.path.exists(impact_model_path):
                logger.info(f"Loading impact assessment model from {impact_model_path}")
                self.models["impact_assessor"] = load_model(impact_model_path)
            else:
                logger.warning(f"Impact assessment model not found at {impact_model_path}")
                self.models["impact_assessor"] = None
            
            # Recommendation model
            recommendation_model_path = os.path.join(self.model_path, "recommendation_generator.h5")
            if os.path.exists(recommendation_model_path):
                logger.info(f"Loading recommendation model from {recommendation_model_path}")
                self.models["recommendation_generator"] = load_model(recommendation_model_path)
            else:
                logger.warning(f"Recommendation model not found at {recommendation_model_path}")
                self.models["recommendation_generator"] = None
        
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    def _load_knowledge_base(self):
        """Load the knowledge base of causes and recommendations"""
        try:
            # Knowledge base path
            kb_path = os.path.join(self.data_path, "knowledge_base.json")
            
            if os.path.exists(kb_path):
                logger.info(f"Loading knowledge base from {kb_path}")
                with open(kb_path, 'r') as f:
                    self.knowledge_base = json.load(f)
            else:
                logger.warning(f"Knowledge base not found at {kb_path}, using default")
                # Default knowledge base (simplified)
                self.knowledge_base = {
                    "causes": {
                        "congestion": [
                            "High traffic volume exceeding capacity",
                            "Road construction or maintenance",
                            "Lane closure or restriction",
                            "Special event generating unusual traffic",
                            "Weather conditions affecting driver behavior",
                            "Seasonal traffic patterns",
                            "Peak hour demand",
                            "Public transportation disruption",
                            "Navigation app routing concentration"
                        ],
                        "incident": [
                            "Vehicle collision",
                            "Vehicle breakdown",
                            "Debris on roadway",
                            "Traffic signal malfunction",
                            "Emergency vehicle activity",
                            "Hazardous material spill",
                            "Police activity",
                            "Pedestrian-related incident",
                            "Weather-related hazard"
                        ],
                        "unusual_flow": [
                            "Diversion from nearby congestion",
                            "Alternative route selection by drivers",
                            "Navigation app influence",
                            "Special event nearby",
                            "Road closure on connected routes",
                            "Public transportation changes",
                            "New traffic pattern emergence",
                            "Temporary road configuration",
                            "Traffic signal timing changes"
                        ],
                        "anomaly": [
                            "Sensor malfunction",
                            "Data processing issue",
                            "Communication disruption",
                            "System calibration problem",
                            "Environmental interference",
                            "Power fluctuation",
                            "Software error",
                            "Unexpected traffic behavior",
                            "External system interference"
                        ]
                    },
                    "recommendations": {
                        "congestion": {
                            "low": [
                                "Monitor traffic conditions",
                                "Prepare for potential intervention",
                                "Alert traffic operators"
                            ],
                            "medium": [
                                "Adjust signal timing",
                                "Issue traffic advisory",
                                "Activate variable message signs",
                                "Implement minor rerouting"
                            ],
                            "high": [
                                "Deploy traffic management personnel",
                                "Implement congestion management plan",
                                "Coordinate multi-intersection signal timing",
                                "Issue traveler information advisories",
                                "Activate alternate route plans"
                            ],
                            "critical": [
                                "Activate emergency traffic management",
                                "Deploy all available resources",
                                "Coordinate with emergency services",
                                "Implement major rerouting strategies",
                                "Issue emergency traveler information",
                                "Request additional personnel"
                            ]
                        },
                        "incident": {
                            "low": [
                                "Verify incident via cameras",
                                "Monitor for resolution",
                                "Prepare response if escalation occurs"
                            ],
                            "medium": [
                                "Dispatch incident response unit",
                                "Adjust signal timing at affected intersections",
                                "Issue localized traffic advisory",
                                "Monitor incident progression"
                            ],
                            "high": [
                                "Dispatch emergency services",
                                "Implement incident management protocol",
                                "Adjust surrounding traffic control",
                                "Issue area-wide advisories",
                                "Activate alternate route signage"
                            ],
                            "critical": [
                                "Activate full emergency response",
                                "Coordinate police, fire, and medical services",
                                "Implement emergency traffic management",
                                "Close affected roadways if necessary",
                                "Issue emergency public notifications",
                                "Activate mutual aid agreements"
                            ]
                        },
                        "unusual_flow": {
                            "low": [
                                "Monitor pattern development",
                                "Identify potential causes",
                                "Document for future analysis"
                            ],
                            "medium": [
                                "Check for correlated events",
                                "Adjust affected signal timing",
                                "Monitor for pattern stabilization",
                                "Prepare contingency measures"
                            ],
                            "high": [
                                "Implement adaptive traffic management",
                                "Coordinate with regional traffic centers",
                                "Deploy targeted resources",
                                "Issue advisories for affected areas",
                                "Monitor secondary effects"
                            ],
                            "critical": [
                                "Activate emergency flow management",
                                "Deploy traffic personnel to key locations",
                                "Implement system-wide adjustments",
                                "Coordinate multi-agency response",
                                "Issue widespread traffic advisories"
                            ]
                        },
                        "anomaly": {
                            "low": [
                                "Log for data quality review",
                                "Schedule routine maintenance check",
                                "Monitor for pattern recognition"
                            ],
                            "medium": [
                                "Verify sensor functionality",
                                "Check system configuration",
                                "Review recent maintenance activities",
                                "Test data processing pipeline"
                            ],
                            "high": [
                                "Dispatch technical personnel",
                                "Implement backup data sources",
                                "Run system diagnostics",
                                "Review network connectivity",
                                "Check for cyber security issues"
                            ],
                            "critical": [
                                "Activate emergency technical response",
                                "Switch to backup systems",
                                "Implement manual traffic monitoring",
                                "Conduct full system diagnostics",
                                "Deploy emergency maintenance team"
                            ]
                        }
                    }
                }
        
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            # Minimal fallback
            self.knowledge_base = {
                "causes": {anomaly_type: [] for anomaly_type in self.ANOMALY_TYPES},
                "recommendations": {anomaly_type: {severity: [] for severity in self.SEVERITY_LEVELS} for anomaly_type in self.ANOMALY_TYPES}
            }
    
    def _extract_features(self, anomaly: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from anomaly data for model input
        
        Args:
            anomaly: Dict containing anomaly data
            
        Returns:
            Feature vector as numpy array
        """
        # Extract relevant fields
        features = []
        
        # Anomaly score
        features.append(anomaly.get("anomaly_score", 0.5))
        
        # Anomaly type as one-hot encoding
        type_one_hot = [0] * len(self.ANOMALY_TYPES)
        anomaly_type = anomaly.get("anomaly_type", "anomaly")
        if anomaly_type in self.ANOMALY_TYPES:
            type_index = self.ANOMALY_TYPES.index(anomaly_type)
            type_one_hot[type_index] = 1
        features.extend(type_one_hot)
        
        # Severity as one-hot encoding
        severity_one_hot = [0] * len(self.SEVERITY_LEVELS)
        severity = anomaly.get("anomaly_severity", "medium")
        if severity in self.SEVERITY_LEVELS:
            severity_index = self.SEVERITY_LEVELS.index(severity)
            severity_one_hot[severity_index] = 1
        features.extend(severity_one_hot)
        
        # Traffic metrics
        measured_values = anomaly.get("measured_values", {})
        
        # Get common traffic metrics
        features.append(measured_values.get("speed", 0))
        features.append(measured_values.get("volume", 0))
        features.append(measured_values.get("occupancy", 0))
        features.append(measured_values.get("density", 0))
        features.append(measured_values.get("speed_std", 0))
        features.append(measured_values.get("congestion_level", 0))
        
        # Time features
        timestamp = anomaly.get("timestamp", datetime.now().isoformat())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Hour of day (0-23)
        features.append(timestamp.hour)
        
        # Day of week (0-6)
        features.append(timestamp.weekday())
        
        # Is weekend
        features.append(1 if timestamp.weekday() >= 5 else 0)
        
        # Is rush hour (7-9 AM or 4-7 PM)
        is_rush_hour = (7 <= timestamp.hour < 9) or (16 <= timestamp.hour < 19)
        features.append(1 if is_rush_hour else 0)
        
        return np.array(features, dtype=np.float32)
    
    def _classify_causes(self, anomaly: Dict[str, Any], features: np.ndarray) -> List[str]:
        """
        Identify the most likely causes of the anomaly
        
        Args:
            anomaly: Dict containing anomaly data
            features: Extracted feature vector
            
        Returns:
            List of likely causes with probabilities
        """
        anomaly_type = anomaly.get("anomaly_type", "anomaly")
        
        # Get causes from knowledge base for this type
        potential_causes = self.knowledge_base["causes"].get(anomaly_type, [])
        
        # If no causes found or empty, use general anomaly causes
        if not potential_causes:
            potential_causes = self.knowledge_base["causes"].get("anomaly", [])
        
        # If still empty, return default
        if not potential_causes:
            return ["Unknown cause"]
        
        # Rule-based cause selection (when model is not available)
        if not self.models.get("cause_classifier"):
            # Select 2-4 causes based on anomaly severity
            severity = anomaly.get("anomaly_severity", "medium")
            num_causes = {
                "low": 2,
                "medium": 2,
                "high": 3,
                "critical": 4
            }.get(severity, 2)
            
            # Select random causes (in a real system, this would use heuristics)
            import random
            selected_causes = random.sample(
                potential_causes, 
                min(num_causes, len(potential_causes))
            )
            
            return selected_causes
        
        # Use ML model for cause classification (simulated)
        # In a real system, this would pass features to the model
        
        # Simulated model output
        cause_probs = np.random.random(len(potential_causes))
        cause_probs = cause_probs / cause_probs.sum()  # Normalize
        
        # Select top causes
        num_causes = min(3, len(potential_causes))
        top_indices = np.argsort(cause_probs)[-num_causes:]
        
        # Return top causes
        return [potential_causes[i] for i in top_indices]
    
    def _generate_recommendations(self, anomaly: Dict[str, Any], causes: List[str]) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            anomaly: Dict containing anomaly data
            causes: List of identified causes
            
        Returns:
            List of recommendations
        """
        anomaly_type = anomaly.get("anomaly_type", "anomaly")
        severity = anomaly.get("anomaly_severity", "medium")
        
        # Get recommendations from knowledge base for this type and severity
        type_recommendations = self.knowledge_base["recommendations"].get(anomaly_type, {})
        potential_recommendations = type_recommendations.get(severity, [])
        
        # If no recommendations found, try general recommendations
        if not potential_recommendations:
            general_recommendations = self.knowledge_base["recommendations"].get("anomaly", {})
            potential_recommendations = general_recommendations.get(severity, [])
        
        # If still empty, return default
        if not potential_recommendations:
            return ["Monitor the situation"]
        
        # Rule-based recommendation selection (when model is not available)
        if not self.models.get("recommendation_generator"):
            # Select 2-4 recommendations based on severity
            num_recommendations = {
                "low": 2,
                "medium": 3,
                "high": 3,
                "critical": 4
            }.get(severity, 2)
            
            # Select random recommendations (in a real system, this would use heuristics)
            import random
            selected_recommendations = random.sample(
                potential_recommendations, 
                min(num_recommendations, len(potential_recommendations))
            )
            
            return selected_recommendations
        
        # Use ML model for recommendation generation (simulated)
        # In a real system, this would use the model to generate recommendations
        
        # Simulated model output
        num_recommendations = min(3, len(potential_recommendations))
        recommendation_indices = np.random.choice(
            len(potential_recommendations), 
            size=num_recommendations, 
            replace=False
        )
        
        # Return selected recommendations
        return [potential_recommendations[i] for i in recommendation_indices]
    
    def _generate_description(self, anomaly: Dict[str, Any], causes: List[str]) -> str:
        """
        Generate a natural language description of the anomaly
        
        Args:
            anomaly: Dict containing anomaly data
            causes: List of identified causes
            
        Returns:
            Natural language description
        """
        anomaly_type = anomaly.get("anomaly_type", "anomaly").replace("_", " ")
        severity = anomaly.get("anomaly_severity", "medium")
        score = anomaly.get("anomaly_score", 0.5)
        
        # Format timestamp
        timestamp = anomaly.get("timestamp", datetime.now().isoformat())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Get relevant metrics
        measured_values = anomaly.get("measured_values", {})
        speed = measured_values.get("speed", 0)
        volume = measured_values.get("volume", 0)
        congestion = measured_values.get("congestion_level", 0)
        
        # Description templates based on anomaly type
        templates = {
            "congestion": [
                f"A {severity} traffic congestion anomaly detected at {time_str} with an anomaly score of {score:.2f}. ",
                f"Traffic speed is {speed:.1f} km/h with volume of {volume} vehicles and congestion level of {congestion:.2f}. ",
                f"This indicates an unusual congestion pattern that doesn't match typical traffic behavior for this time and location."
            ],
            "incident": [
                f"A {severity} traffic incident anomaly detected at {time_str} with an anomaly score of {score:.2f}. ",
                f"Traffic metrics show speed at {speed:.1f} km/h with abnormal flow patterns. ",
                f"This pattern is consistent with a traffic incident causing disruption to normal traffic flow."
            ],
            "unusual flow": [
                f"An unusual traffic flow pattern (severity: {severity}) detected at {time_str} with an anomaly score of {score:.2f}. ",
                f"Traffic volume is {volume} vehicles with atypical distribution across lanes and directions. ",
                f"This represents a significant deviation from expected traffic patterns for this location and time."
            ],
            "anomaly": [
                f"A general traffic anomaly (severity: {severity}) detected at {time_str} with an anomaly score of {score:.2f}. ",
                f"Multiple traffic metrics including speed ({speed:.1f} km/h) and volume ({volume} vehicles) show unusual values. ",
                f"This pattern doesn't match any typical traffic behavior and requires investigation."
            ]
        }
        
        # Get template for this type, or default to general anomaly
        template_parts = templates.get(anomaly_type, templates["anomaly"])
        
        # Join template parts
        description = "".join(template_parts)
        
        # Add cause information if available
        if causes:
            cause_str = ", ".join(causes[:-1])
            if len(causes) > 1:
                cause_str += f", and {causes[-1]}"
            else:
                cause_str = causes[0]
            
            description += f" Likely causes include: {cause_str}."
        
        return description
    
    async def analyze_anomaly(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a traffic anomaly
        
        Args:
            anomaly: Dict containing anomaly data
            
        Returns:
            Dict with analysis results
        """
        try:
            # Extract features
            features = self._extract_features(anomaly)
            
            # Identify causes
            causes = self._classify_causes(anomaly, features)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(anomaly, causes)
            
            # Generate description
            description = self._generate_description(anomaly, causes)
            
            # Calculate confidence
            confidence = min(1.0, max(0.5, anomaly.get("anomaly_score", 0.5) * 1.1))
            
            # Construct analysis results
            analysis = {
                "description": description,
                "possibleCauses": causes,
                "recommendations": recommendations,
                "confidence": confidence
            }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing anomaly: {str(e)}")
            # Return basic analysis in case of error
            return {
                "description": "Analysis could not be completed due to a system error.",
                "possibleCauses": ["Unknown cause due to analysis failure"],
                "recommendations": ["Monitor the anomaly manually"],
                "confidence": 0.5
            }
    
    async def analyze_anomaly_with_llm(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform anomaly analysis with the help of a language model
        
        Args:
            anomaly: Dict containing anomaly data
            
        Returns:
            Dict with analysis results
        """
        if not self.use_llm:
            # Fall back to regular analysis if LLM is not enabled
            return await self.analyze_anomaly(anomaly)
        
        try:
            # Construct prompt for the LLM
            anomaly_type = anomaly.get("anomaly_type", "anomaly").replace("_", " ")
            severity = anomaly.get("anomaly_severity", "medium")
            score = anomaly.get("anomaly_score", 0.5)
            
            # Format measured values
            measured_values = anomaly.get("measured_values", {})
            measured_str = ", ".join([f"{k}: {v}" for k, v in measured_values.items()])
            
            prompt = f"""
            Analyze the following traffic anomaly:
            - Type: {anomaly_type}
            - Severity: {severity}
            - Score: {score}
            - Metrics: {measured_str}
            
            Provide:
            1. A detailed description of what this anomaly means
            2. A list of possible causes
            3. Specific recommendations for traffic operators
            """
            
            # Call LLM service (simulated)
            # In a real system, this would make an API call to a language model service
            
            # Simulate LLM thinking time
            await asyncio.sleep(2)
            
            # Generate a basic analysis (simulating LLM output)
            basic_analysis = await self.analyze_anomaly(anomaly)
            
            # Add a bit more detail to make it seem like LLM output
            description = basic_analysis["description"]
            causes = basic_analysis["possibleCauses"]
            recommendations = basic_analysis["recommendations"]
            
            # Enhance description
            description += " The pattern analysis indicates a significant deviation from baseline behavior that requires attention based on both current metrics and historical patterns for this location and time of day."
            
            # Enhance recommendations
            enhanced_recommendations = []
            for rec in recommendations:
                enhanced_recommendations.append(f"{rec} to minimize impact on the transportation network.")
            
            # Return enhanced analysis
            return {
                "description": description,
                "possibleCauses": causes,
                "recommendations": enhanced_recommendations,
                "confidence": basic_analysis["confidence"]
            }
        
        except Exception as e:
            logger.error(f"Error using LLM for anomaly analysis: {str(e)}")
            # Fall back to regular analysis
            return await self.analyze_anomaly(anomaly)


# Create a singleton instance
analysis_service = AIAnalysisService()


async def analyze_anomaly(anomaly: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a traffic anomaly using AI
    
    Args:
        anomaly: Dict containing anomaly data
        
    Returns:
        Dict with analysis results
    """
    # Use the singleton service instance
    return await analysis_service.analyze_anomaly(anomaly)


if __name__ == "__main__":
    # Test with a sample anomaly
    import asyncio
    
    sample_anomaly = {
        "id": "test-123",
        "timestamp": datetime.now().isoformat(),
        "location_id": "L001",
        "anomaly_score": 0.85,
        "anomaly_type": "congestion",
        "anomaly_severity": "high",
        "measured_values": {
            "speed": 15.5,
            "volume": 480,
            "occupancy": 85.0,
            "density": 31.0,
            "speed_std": 8.5,
            "congestion_level": 0.75
        },
        "latitude": 40.7128,
        "longitude": -74.0060,
        "detection_time": datetime.now().isoformat()
    }
    
    async def test():
        result = await analyze_anomaly(sample_anomaly)
        print("Analysis Result:")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())