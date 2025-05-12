"""
Traffic Anomaly Detection Service

This module provides real-time anomaly detection for traffic patterns using machine learning
techniques including time series analysis, clustering, and deep learning approaches.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Kafka for event streaming
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("traffic-anomaly-detection")


class AnomalyDetectionConfig:
    """Configuration for the anomaly detection service"""
    
    def __init__(self):
        # Service configuration
        self.model_path = os.environ.get("MODEL_PATH", "/models/anomaly")
        self.data_path = os.environ.get("DATA_PATH", "/data/traffic")
        self.debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        
        # Kafka configuration
        self.kafka_bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.kafka_input_topic = os.environ.get("KAFKA_INPUT_TOPIC", "traffic-data-stream")
        self.kafka_output_topic = os.environ.get("KAFKA_OUTPUT_TOPIC", "traffic-anomalies")
        self.kafka_group_id = os.environ.get("KAFKA_GROUP_ID", "anomaly-detection-service")
        
        # Model hyperparameters
        self.window_size = int(os.environ.get("WINDOW_SIZE", "24"))
        self.batch_size = int(os.environ.get("BATCH_SIZE", "32"))
        self.prediction_horizon = int(os.environ.get("PREDICTION_HORIZON", "12"))
        self.detection_threshold = float(os.environ.get("DETECTION_THRESHOLD", "0.85"))
        
        # Set logging level
        if hasattr(logging, self.log_level):
            logger.setLevel(getattr(logging, self.log_level))


class TrafficAnomalyDetector:
    """
    Anomaly detector for traffic patterns based on multiple detection algorithms.
    
    This class implements several approaches for anomaly detection:
    1. Statistical methods (z-score, IQR)
    2. Machine learning (Isolation Forest, DBSCAN)
    3. Deep learning (autoencoder, LSTM)
    
    The detector can work in different modes:
    - Statistical: Fast, but less accurate for complex patterns
    - ML: Good balance between speed and accuracy
    - Deep: Most accurate but computationally intensive
    - Ensemble: Combines multiple methods for higher confidence
    """
    
    MODEL_TYPES = ["statistical", "isolation_forest", "dbscan", "autoencoder", "lstm", "ensemble"]
    
    def __init__(self, config: AnomalyDetectionConfig):
        """Initialize the anomaly detector with the given configuration"""
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.detection_mode = os.environ.get("DETECTION_MODE", "ensemble")
        
        # Validate detection mode
        if self.detection_mode not in self.MODEL_TYPES:
            logger.warning(
                f"Invalid detection mode '{self.detection_mode}'. "
                f"Falling back to 'ensemble'."
            )
            self.detection_mode = "ensemble"
        
        # Initialize models based on mode
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the appropriate models based on the detection mode"""
        if self.detection_mode in ["statistical", "ensemble"]:
            self.models["statistical"] = {"name": "statistical"}
        
        if self.detection_mode in ["isolation_forest", "ensemble"]:
            self.models["isolation_forest"] = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            )
        
        if self.detection_mode in ["dbscan", "ensemble"]:
            self.models["dbscan"] = DBSCAN(
                eps=0.5,
                min_samples=5,
                n_jobs=-1
            )
        
        if self.detection_mode in ["autoencoder", "ensemble"]:
            # Load or create autoencoder model
            model_path = os.path.join(self.config.model_path, "autoencoder_model.h5")
            if os.path.exists(model_path):
                logger.info(f"Loading autoencoder model from {model_path}")
                self.models["autoencoder"] = load_model(model_path)
            else:
                logger.info("Autoencoder model not found, will be created during training")
                self.models["autoencoder"] = None
        
        if self.detection_mode in ["lstm", "ensemble"]:
            # Load or create LSTM model
            model_path = os.path.join(self.config.model_path, "lstm_model.h5")
            if os.path.exists(model_path):
                logger.info(f"Loading LSTM model from {model_path}")
                self.models["lstm"] = load_model(model_path)
            else:
                logger.info("LSTM model not found, will be created during training")
                self.models["lstm"] = None
    
    def _create_autoencoder_model(self, input_dim: int) -> Model:
        """
        Create a deep autoencoder model for anomaly detection
        
        Args:
            input_dim: The dimensionality of the input features
            
        Returns:
            A compiled Keras autoencoder model
        """
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Create an LSTM-based model for time series anomaly detection
        
        Args:
            input_shape: A tuple (sequence_length, num_features)
            
        Returns:
            A compiled Keras LSTM model
        """
        # Sequence input
        input_layer = Input(shape=input_shape)
        
        # LSTM layers
        x = LSTM(128, return_sequences=True)(input_layer)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        # Reconstruction layers
        x = Dense(64, activation='relu')(x)
        x = Dense(input_shape[1] * input_shape[0], activation='linear')(x)
        
        # Reshape to match input dimensions
        output_layer = tf.keras.layers.Reshape(input_shape)(x)
        
        # Model
        lstm_model = Model(input_layer, output_layer)
        lstm_model.compile(optimizer='adam', loss='mse')
        
        return lstm_model
    
    def train(self, historical_data: pd.DataFrame) -> None:
        """
        Train the anomaly detection models using historical traffic data
        
        Args:
            historical_data: DataFrame containing historical traffic metrics
                Expected columns include 'timestamp', 'location_id', 'speed',
                'volume', 'occupancy', etc.
        """
        logger.info(f"Training anomaly detection models with {len(historical_data)} samples")
        
        # Prepare features
        feature_columns = self._extract_feature_columns(historical_data)
        X = historical_data[feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the active models
        if "isolation_forest" in self.models:
            logger.info("Training Isolation Forest model")
            self.models["isolation_forest"].fit(X_scaled)
        
        if "dbscan" in self.models:
            logger.info("Training DBSCAN model")
            self.models["dbscan"].fit(X_scaled)
        
        if "autoencoder" in self.models or self.detection_mode in ["autoencoder", "ensemble"]:
            logger.info("Training Autoencoder model")
            autoencoder = self._create_autoencoder_model(X_scaled.shape[1])
            
            # Train autoencoder
            checkpoint_path = os.path.join(self.config.model_path, "autoencoder_checkpoint.h5")
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(checkpoint_path, save_best_only=True)
            ]
            
            autoencoder.fit(
                X_scaled, X_scaled,
                epochs=50,
                batch_size=self.config.batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1 if self.config.debug_mode else 2
            )
            
            # Save model
            model_path = os.path.join(self.config.model_path, "autoencoder_model.h5")
            autoencoder.save(model_path)
            self.models["autoencoder"] = autoencoder
        
        if "lstm" in self.models or self.detection_mode in ["lstm", "ensemble"]:
            logger.info("Training LSTM model")
            
            # Prepare sequential data
            X_sequences = self._create_sequences(X_scaled, self.config.window_size)
            
            # Create LSTM model
            lstm_model = self._create_lstm_model((self.config.window_size, X_scaled.shape[1]))
            
            # Train LSTM model
            checkpoint_path = os.path.join(self.config.model_path, "lstm_checkpoint.h5")
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(checkpoint_path, save_best_only=True)
            ]
            
            lstm_model.fit(
                X_sequences, X_sequences,
                epochs=50,
                batch_size=self.config.batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1 if self.config.debug_mode else 2
            )
            
            # Save model
            model_path = os.path.join(self.config.model_path, "lstm_model.h5")
            lstm_model.save(model_path)
            self.models["lstm"] = lstm_model
        
        self.is_trained = True
        logger.info("Training completed successfully")
    
    def _extract_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Extract the relevant feature columns for anomaly detection"""
        # Exclude non-feature columns like timestamps, IDs, etc.
        exclude_columns = ['timestamp', 'id', 'location_id', 'anomaly', 'created_at', 'updated_at']
        
        # Get numerical feature columns
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        return feature_columns
    
    def _create_sequences(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Create sequences for time series models"""
        sequences = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i+window_size])
        return np.array(sequences)
    
    def detect_anomalies(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in current traffic data
        
        Args:
            current_data: DataFrame containing current traffic metrics to evaluate
        
        Returns:
            DataFrame with original data plus anomaly scores and detection flags
        """
        if not self.is_trained:
            logger.warning("Models have not been trained. Anomaly detection may be inaccurate.")
        
        # Prepare features
        feature_columns = self._extract_feature_columns(current_data)
        X = current_data[feature_columns].values
        
        # Scale features using the same scaler used during training
        X_scaled = self.scaler.transform(X)
        
        # Initialize scores
        anomaly_scores = {model_name: np.zeros(len(X)) for model_name in self.models}
        
        # Calculate anomaly scores for each model
        if "statistical" in self.models:
            logger.debug("Computing statistical anomaly scores")
            anomaly_scores["statistical"] = self._statistical_anomaly_score(X_scaled)
        
        if "isolation_forest" in self.models:
            logger.debug("Computing Isolation Forest anomaly scores")
            # Convert raw scores to positive anomaly scores
            raw_scores = -self.models["isolation_forest"].decision_function(X_scaled)
            anomaly_scores["isolation_forest"] = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        
        if "dbscan" in self.models:
            logger.debug("Computing DBSCAN anomaly scores")
            # Points labeled as -1 are anomalies
            labels = self.models["dbscan"].fit_predict(X_scaled)
            anomaly_scores["dbscan"] = np.array([1.0 if label == -1 else 0.0 for label in labels])
        
        if "autoencoder" in self.models:
            logger.debug("Computing Autoencoder anomaly scores")
            # Reconstruction error as anomaly score
            reconstructions = self.models["autoencoder"].predict(X_scaled)
            mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
            anomaly_scores["autoencoder"] = (mse - mse.min()) / (mse.max() - mse.min()) if mse.max() > mse.min() else mse
        
        if "lstm" in self.models:
            logger.debug("Computing LSTM anomaly scores")
            # Prepare sequential data
            # For simplicity with a single point, we repeat it to create a sequence
            X_sequences = np.array([np.tile(x, (self.config.window_size, 1)) for x in X_scaled])
            
            # Reconstruction error as anomaly score
            reconstructions = self.models["lstm"].predict(X_sequences)
            mse = np.mean(np.square(X_sequences - reconstructions), axis=(1, 2))
            anomaly_scores["lstm"] = (mse - mse.min()) / (mse.max() - mse.min()) if mse.max() > mse.min() else mse
        
        # Combine scores based on detection mode
        if self.detection_mode == "ensemble":
            logger.debug("Computing ensemble anomaly scores")
            # Average the scores from all models
            combined_scores = np.zeros(len(X))
            for model_name, scores in anomaly_scores.items():
                combined_scores += scores
            
            combined_scores /= len(anomaly_scores)
            final_scores = combined_scores
        else:
            # Use the score from the specified model
            final_scores = anomaly_scores[self.detection_mode]
        
        # Add results to DataFrame
        result_df = current_data.copy()
        result_df["anomaly_score"] = final_scores
        result_df["is_anomaly"] = final_scores > self.config.detection_threshold
        
        # Additional anomaly details
        result_df["anomaly_type"] = result_df.apply(self._classify_anomaly_type, axis=1)
        result_df["anomaly_severity"] = result_df.apply(self._calculate_anomaly_severity, axis=1)
        
        return result_df
    
    def _statistical_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores using statistical methods"""
        # Z-score method
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-10))
        
        # Average across features
        avg_z_scores = np.mean(z_scores, axis=1)
        
        # Normalize to 0-1 range
        normalized_scores = (avg_z_scores - avg_z_scores.min()) / (avg_z_scores.max() - avg_z_scores.min() + 1e-10)
        
        return normalized_scores
    
    def _classify_anomaly_type(self, row: pd.Series) -> str:
        """Classify the type of anomaly based on the feature values"""
        if not row["is_anomaly"]:
            return "normal"
        
        # Check for congestion anomaly
        if "speed" in row and "volume" in row:
            if row["speed"] < 20 and row["volume"] > 80:
                return "congestion"
        
        # Check for incident anomaly
        if "speed" in row and "speed_std" in row:
            if row["speed"] < 30 and row["speed_std"] > 15:
                return "incident"
        
        # Check for flow anomaly
        if "volume" in row and "expected_volume" in row:
            if abs(row["volume"] - row["expected_volume"]) > 30:
                return "unusual_flow"
        
        # Default to generic anomaly
        return "anomaly"
    
    def _calculate_anomaly_severity(self, row: pd.Series) -> str:
        """Calculate the severity of the anomaly"""
        if not row["is_anomaly"]:
            return "none"
        
        score = row["anomaly_score"]
        
        if score > 0.95:
            return "critical"
        elif score > 0.9:
            return "high"
        elif score > 0.8:
            return "medium"
        else:
            return "low"


class TrafficAnomalyProcessor:
    """
    Process traffic data streams for anomaly detection
    
    This class handles the following:
    1. Consuming traffic data from Kafka
    2. Running anomaly detection algorithms
    3. Publishing detected anomalies back to Kafka
    4. Maintaining state for context-aware detection
    """
    
    def __init__(self, config: AnomalyDetectionConfig):
        """Initialize the anomaly processor with the given configuration"""
        self.config = config
        self.detector = TrafficAnomalyDetector(config)
        self.is_running = False
        self.recent_data = pd.DataFrame()
        self.historical_data_loaded = False
        
        # Initialize Kafka consumer and producer
        self._initialize_kafka()
    
    def _initialize_kafka(self):
        """Initialize Kafka consumer and producer"""
        # Consumer configuration
        self.consumer_config = {
            'bootstrap.servers': self.config.kafka_bootstrap_servers,
            'group.id': self.config.kafka_group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        }
        
        # Producer configuration
        self.producer_config = {
            'bootstrap.servers': self.config.kafka_bootstrap_servers,
        }
        
        # Create consumer and producer
        self.consumer = Consumer(self.consumer_config)
        self.producer = Producer(self.producer_config)
        
        # Subscribe to input topic
        self.consumer.subscribe([self.config.kafka_input_topic])
    
    def load_historical_data(self) -> bool:
        """
        Load historical traffic data for model training
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Data file path
            data_file = os.path.join(self.config.data_path, "historical_traffic_data.csv")
            
            if os.path.exists(data_file):
                logger.info(f"Loading historical data from {data_file}")
                data = pd.read_csv(data_file)
                
                # Parse timestamps
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                logger.info(f"Loaded {len(data)} historical records")
                
                # Train the detector with the historical data
                self.detector.train(data)
                
                # Store some recent data for context
                recent_cutoff = datetime.now() - timedelta(days=1)
                if 'timestamp' in data.columns:
                    self.recent_data = data[data['timestamp'] >= recent_cutoff].copy()
                else:
                    # If no timestamp, just take the last N rows
                    self.recent_data = data.tail(1000).copy()
                
                self.historical_data_loaded = True
                return True
            else:
                logger.warning(f"Historical data file not found: {data_file}")
                return False
        
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return False
    
    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single traffic data message
        
        Args:
            message: Dict containing traffic data
        
        Returns:
            Dict with anomaly results if anomaly detected, None otherwise
        """
        try:
            # Convert message to DataFrame
            message_df = pd.DataFrame([message])
            
            # Add timestamp if missing
            if 'timestamp' not in message_df.columns:
                message_df['timestamp'] = datetime.now().isoformat()
            
            # Detect anomalies
            result_df = self.detector.detect_anomalies(message_df)
            
            # Add to recent data for context
            self.recent_data = pd.concat([self.recent_data, message_df]).tail(1000)
            
            # Return anomaly results if an anomaly was detected
            if result_df["is_anomaly"].any():
                # Get the first (and only) row from the result
                anomaly_row = result_df.iloc[0]
                
                # Create anomaly result message
                result = {
                    "timestamp": anomaly_row.get("timestamp", datetime.now().isoformat()),
                    "location_id": anomaly_row.get("location_id", "unknown"),
                    "anomaly_score": float(anomaly_row["anomaly_score"]),
                    "anomaly_type": anomaly_row["anomaly_type"],
                    "anomaly_severity": anomaly_row["anomaly_severity"],
                    "measured_values": {
                        col: float(anomaly_row[col]) 
                        for col in self.detector._extract_feature_columns(result_df) 
                        if col in anomaly_row and not pd.isna(anomaly_row[col])
                    },
                    "detection_time": datetime.now().isoformat()
                }
                
                return result
            
            return None
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return None
    
    def delivery_report(self, err, msg):
        """Delivery callback for Kafka producer"""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def publish_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """
        Publish anomaly to Kafka output topic
        
        Args:
            anomaly: Dict containing anomaly details
        """
        try:
            # Convert to JSON string
            anomaly_json = json.dumps(anomaly)
            
            # Produce to Kafka
            self.producer.produce(
                self.config.kafka_output_topic,
                anomaly_json.encode('utf-8'),
                callback=self.delivery_report
            )
            
            # Flush to ensure message is sent
            self.producer.flush()
            
            logger.info(f"Published anomaly: {anomaly['anomaly_type']} with severity {anomaly['anomaly_severity']}")
        
        except Exception as e:
            logger.error(f"Error publishing anomaly: {str(e)}")
    
    def start_processing(self) -> None:
        """Start processing traffic data for anomalies"""
        # Load historical data if not already loaded
        if not self.historical_data_loaded:
            success = self.load_historical_data()
            if not success:
                logger.warning("Failed to load historical data. Starting with untrained models.")
        
        self.is_running = True
        logger.info(f"Starting traffic anomaly detection on topic {self.config.kafka_input_topic}")
        
        try:
            while self.is_running:
                msg = self.consumer.poll(1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"Reached end of partition {msg.topic()} [{msg.partition()}]")
                    else:
                        logger.error(f"Error while consuming: {msg.error()}")
                    continue
                
                try:
                    # Parse message
                    message_json = msg.value().decode('utf-8')
                    message = json.loads(message_json)
                    
                    # Process message
                    anomaly = self.process_message(message)
                    
                    # Publish anomaly if detected
                    if anomaly:
                        self.publish_anomaly(anomaly)
                    
                    # Commit offset
                    self.consumer.commit(msg)
                
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    continue
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Stopping...")
        
        finally:
            # Close consumer and producer
            self.consumer.close()
            self.is_running = False
            logger.info("Traffic anomaly detection stopped")


def main():
    """Main entry point for the anomaly detection service"""
    try:
        logger.info("Starting Traffic Anomaly Detection Service")
        
        # Initialize configuration
        config = AnomalyDetectionConfig()
        
        # Create processor
        processor = TrafficAnomalyProcessor(config)
        
        # Start processing
        processor.start_processing()
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())