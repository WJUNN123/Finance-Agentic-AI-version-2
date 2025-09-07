"""
Price forecasting module using Prophet and LSTM models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

class Forecaster:
    """Handles cryptocurrency price forecasting using multiple models."""
    
    def __init__(self):
        self.prophet_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        
        self.lstm_params = {
            'window_size': 30,
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2
        }
    
    def generate_forecasts(self, price_history: pd.DataFrame, 
                         horizon_days: int = 7) -> Dict:
        """
        Generate price forecasts using available models.
        
        Args:
            price_history: DataFrame with price data and datetime index
            horizon_days: Number of days to forecast
            
        Returns:
            Dictionary containing forecast results
        """
        results = {
            'prophet_forecast': None,
            'lstm_forecast': None,
            'ensemble_forecast': None,
            'forecast_dates': [],
            'confidence_intervals': {},
            'model_performance': {}
        }
        
        if price_history.empty or 'price' not in price_history.columns:
            return results
        
        # Generate forecast dates
        last_date = price_history.index[-1]
        forecast_dates = [
            last_date + timedelta(days=i+1) 
            for i in range(horizon_days)
        ]
        results['forecast_dates'] = forecast_dates
        
        # Prophet forecast
        if PROPHET_AVAILABLE:
            try:
                prophet_result = self._prophet_forecast(
                    price_history, horizon_days
                )
                results['prophet_forecast'] = prophet_result
            except Exception as e:
                print(f"Prophet forecast failed: {e}")
        
        # LSTM forecast
        if TENSORFLOW_AVAILABLE:
            try:
                lstm_result = self._lstm_forecast(
                    price_history, horizon_days
                )
                results['lstm_forecast'] = lstm_result
            except Exception as e:
                print(f"LSTM forecast failed: {e}")
        
        # Ensemble forecast
        if results['prophet_forecast'] and results['lstm_forecast']:
            results['ensemble_forecast'] = self._create_ensemble_forecast(
                results['prophet_forecast'], 
                results['lstm_forecast']
            )
        
        return results
    
    def _prophet_forecast(self, price_data: pd.DataFrame, 
                         days: int) -> Dict:
        """Generate Prophet forecast."""
        # Prepare data for Prophet
        df = price_data.reset_index()
        df = df.rename(columns={df.columns[0]: 'ds', 'price': 'y'})
        
        # Initialize and fit Prophet model
        model = Prophet(**self.prophet_params)
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        
        # Extract forecast values
        forecast_values = forecast.tail(days)
        
        return {
            'values': forecast_values['yhat'].tolist(),
            'upper_bound': forecast_values['yhat_upper'].tolist(),
            'lower_bound': forecast_values['yhat_lower'].tolist(),
            'full_forecast': forecast,
            'model': model
        }
    
    def _lstm_forecast(self, price_data: pd.DataFrame, 
                      days: int) -> Dict:
        """Generate LSTM forecast."""
        prices = price_data['price'].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        # Prepare training data
        window_size = self.lstm_params['window_size']
        X, y = self._create_sequences(scaled_prices, window_size)
        
        if len(X) < 10:  # Not enough data
            return None
        
        # Build and train LSTM model
        model = self._build_lstm_model((window_size, 1))
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True
        )
        
        model.fit(
            X, y,
            epochs=self.lstm_params['epochs'],
            batch_size=self.lstm_params['batch_size'],
            validation_split=self.lstm_params['validation_split'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Generate forecasts
        forecast_values = []
        current_batch = scaled_prices[-window_size:].reshape(1, window_size, 1)
        
        for _ in range(days):
            pred = model.predict(current_batch, verbose=0)[0]
            forecast_values.append(pred[0])
            
            # Update batch for next prediction
            current_batch = np.append(
                current_batch[:, 1:, :], 
                pred.reshape(1, 1, 1), 
                axis=1
            )
        
        # Inverse transform predictions
        forecast_values = np.array(forecast_values).reshape(-1, 1)
        forecast_values = scaler.inverse_transform(forecast_values).flatten()
        
        # Calculate confidence intervals (simple approach)
        last_price = prices[-1][0]
        volatility = np.std(np.diff(prices.flatten()))
        confidence_factor = 1.96 * volatility  # 95% confidence
        
        upper_bound = forecast_values + confidence_factor
        lower_bound = forecast_values - confidence_factor
        
        return {
            'values': forecast_values.tolist(),
            'upper_bound': upper_bound.tolist(),
            'lower_bound': lower_bound.tolist(),
            'model': model,
            'scaler': scaler
        }
    
    def _build_lstm_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def _create_sequences(self, data, window_size):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def _create_ensemble_forecast(self, prophet_result: Dict, 
                                lstm_result: Dict) -> Dict:
        """Create ensemble forecast from Prophet and LSTM."""
        if not prophet_result or not lstm_result:
            return None
        
        prophet_values = np.array(prophet_result['values'])
        lstm_values = np.array(lstm_result['values'])
        
        # Weighted average (can be adjusted based on model performance)
        prophet_weight = 0.6
        lstm_weight = 0.4
        
        ensemble_values = (
            prophet_weight * prophet_values + 
            lstm_weight * lstm_values
        )
        
        # Combine confidence intervals
        prophet_upper = np.array(prophet_result['upper_bound'])
        prophet_lower = np.array(prophet_result['lower_bound'])
        lstm_upper = np.array(lstm_result['upper_bound'])
        lstm_lower = np.array(lstm_result['lower_bound'])
        
        ensemble_upper = (
            prophet_weight * prophet_upper + 
            lstm_weight * lstm_upper
        )
        ensemble_lower = (
            prophet_weight * prophet_lower + 
            lstm_weight * lstm_lower
        )
        
        return {
            'values': ensemble_values.tolist(),
            'upper_bound': ensemble_upper.tolist(),
            'lower_bound': ensemble_lower.tolist(),
            'prophet_weight': prophet_weight,
            'lstm_weight': lstm_weight
        }
    
    def calculate_forecast_metrics(self, actual_prices: pd.Series, 
                                 forecast_values: List[float]) -> Dict:
        """Calculate forecast accuracy metrics."""
        if len(actual_prices) != len(forecast_values):
            return {}
        
        actual = np.array(actual_prices)
        forecast = np.array(forecast_values)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - forecast))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        
        # Direction Accuracy
        actual_direction = np.diff(actual) > 0
        forecast_direction = np.diff(forecast) > 0
        direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
        
        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy
        }
    
    def get_forecast_summary(self, forecast_results: Dict) -> str:
        """Generate human-readable forecast summary."""
        if not any(forecast_results.values()):
            return "No forecasts available."
        
        summary_parts = []
        
        if forecast_results.get('ensemble_forecast'):
            values = forecast_results['ensemble_forecast']['values']
            last_forecast = values[-1]
            summary_parts.append(f"Ensemble forecast: ${last_forecast:.2f}")
        
        if forecast_results.get('prophet_forecast'):
            values = forecast_results['prophet_forecast']['values']
            last_forecast = values[-1]
            summary_parts.append(f"Prophet forecast: ${last_forecast:.2f}")
        
        if forecast_results.get('lstm_forecast'):
            values = forecast_results['lstm_forecast']['values']
            last_forecast = values[-1]
            summary_parts.append(f"LSTM forecast: ${last_forecast:.2f}")
        
        return " | ".join(summary_parts)
