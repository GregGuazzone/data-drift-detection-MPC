import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
import argparse
warnings.filterwarnings('ignore')

class LocalDriftDetector:
    def __init__(self, 
                 contamination: float = 0.05,
                 window_size: int = 7):
        self.contamination = contamination
        self.window_size = window_size
        
        # Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        
        # Baseline statistics
        self.baseline_stats = {}
        self.is_fitted = False
        
    def _clean_data(self, data: pd.Series) -> pd.Series:
        """Clean input data more thoroughly"""
        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Replace extremely large values (potential overflow issues)
        large_threshold = 1e10
        data = data.where(data.abs() <= large_threshold, np.nan)
        
        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Ensure all values are finite
        data = data.replace([np.inf, -np.inf], 0)
        
        return data

    def _calculate_basic_features(self, data: pd.Series) -> pd.DataFrame:
        """Calculate basic features for anomaly detection with better error handling"""
        features = pd.DataFrame(index=data.index)
        
        # Original value
        features['value'] = data.values
        
        # Rolling statistics with error handling
        features['rolling_mean'] = data.rolling(window=self.window_size, min_periods=1).mean()
        features['rolling_std'] = data.rolling(window=self.window_size, min_periods=1).std()
        
        # Replace NaN std with a small positive value to avoid division by zero
        features['rolling_std'] = features['rolling_std'].fillna(1.0)
        features['rolling_std'] = features['rolling_std'].replace(0, 1.0)
        
        # Rate of change with better handling
        rate_change = data.pct_change()
        # Cap extreme percentage changes
        rate_change = rate_change.clip(-10, 10)  # Cap at 1000% change
        features['rate_of_change'] = rate_change.fillna(0)
        features['rate_of_change_abs'] = features['rate_of_change'].abs()
        
        # Simple trend (slope over window) with error handling
        def safe_polyfit(x):
            try:
                if len(x) >= 2:
                    return np.polyfit(range(len(x)), x, 1)[0]
                else:
                    return 0
            except:
                return 0
        
        features['trend'] = data.rolling(window=self.window_size, min_periods=2).apply(safe_polyfit).fillna(0)
        
        # Fill any remaining NaN values and replace infinite values
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Final check - ensure all values are finite
        for col in features.columns:
            features[col] = features[col].where(np.isfinite(features[col]), 0)
        
        return features

    def fit(self, data: pd.Series, baseline_split: float = 0.3):
        """
        Fit the model on baseline data with robust error handling
        """
        # Clean input data thoroughly
        data = self._clean_data(data)
        
        # Use first portion as baseline
        split_idx = max(10, int(len(data) * baseline_split))  # Ensure minimum baseline size
        baseline_data = data.iloc[:split_idx]
        
        # Store baseline statistics
        self.baseline_stats = {
            'mean': baseline_data.mean(),
            'std': max(baseline_data.std(), 1.0),  # Ensure std is not zero
            'median': baseline_data.median(),
            'data': baseline_data.values
        }
        
        # Calculate features for baseline
        baseline_features = self._calculate_basic_features(baseline_data)
        
        # Additional cleaning for sklearn
        baseline_features = self._clean_features_for_sklearn(baseline_features)
        
        try:
            # Fit scaler
            baseline_scaled = self.scaler.fit_transform(baseline_features)
            
            # Final check for finite values
            if not np.all(np.isfinite(baseline_scaled)):
                print("Warning: Non-finite values detected, applying final cleaning...")
                baseline_scaled = np.nan_to_num(baseline_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Fit isolation forest
            self.isolation_forest.fit(baseline_scaled)
            
            self.is_fitted = True
            
            print(f"Model fitted on {len(baseline_data)} baseline samples")
            print(f"Baseline period: {baseline_data.index[0].strftime('%Y-%m-%d')} to {baseline_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Baseline stats: mean={self.baseline_stats['mean']:.1f}, std={self.baseline_stats['std']:.1f}")
            
        except Exception as e:
            print(f"Error during model fitting: {e}")
            raise

    def _clean_features_for_sklearn(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean features specifically for sklearn compatibility"""
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove columns with zero variance
        for col in features.columns:
            if features[col].var() <= 1e-10:
                features[col] = features[col] + np.random.normal(0, 1e-6, len(features))
        
        # Final check for finite values
        for col in features.columns:
            features[col] = features[col].where(np.isfinite(features[col]), 0)
        
        return features

    def detect_drift(self, data: pd.Series) -> Dict:
        """
        Detect drift in the data with more conservative statistical drift detection
        
        Returns:
            Dictionary with drift detection results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting drift")
        
        # Clean input data
        data = self._clean_data(data)
        
        # Calculate features
        features = self._calculate_basic_features(data)
        
        # Clean features for sklearn
        features = self._clean_features_for_sklearn(features)
        
        # Scale features and detect anomalies
        features_scaled = self.scaler.transform(features)
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        anomaly_labels = self.isolation_forest.predict(features_scaled)
        
        # Much more conservative statistical drift detection
        statistical_drift = []
        window_size = min(60, max(30, len(data) // 8))  # Larger windows for more stability
        
        # Only check every 7 days instead of every day to reduce noise
        for i in range(window_size, len(data), 7):  # Step by 7 days
            window_data = data.iloc[i-window_size:i]
            
            # Multiple criteria for drift detection
            drift_detected = False
            p_value = 1.0
            
            try:
                #find sources
                # 1. KS test with much stricter threshold
                _, ks_p_value = stats.ks_2samp(self.baseline_stats['data'], window_data.values)
                
                # 2. Mean shift test (t-test)
                from scipy.stats import ttest_ind
                _, ttest_p_value = ttest_ind(self.baseline_stats['data'], window_data.values)
                
                # 3. Variance change test (F-test approximation)
                baseline_var = np.var(self.baseline_stats['data'])
                window_var = np.var(window_data.values)
                var_ratio = max(baseline_var, window_var) / max(min(baseline_var, window_var), 1e-10)
                
                # 4. Magnitude change (median shift)
                baseline_median = self.baseline_stats['median']
                window_median = np.median(window_data.values)
                median_change = abs(window_median - baseline_median) / max(baseline_median, 1)
                
                # Drift detected
                drift_criteria = [
                    ks_p_value < 0.001,
                    ttest_p_value < 0.001,
                    var_ratio > 5.0,
                    median_change > 2.0
                ]
                
                drift_detected = all(drift_criteria)
                p_value = min(ks_p_value, ttest_p_value)
                
            except:
                drift_detected = False
                p_value = 1.0
            
            statistical_drift.append({
                'date': data.index[i],
                'drift_detected': drift_detected,
                'p_value': p_value
            })
        
        # Combine results
        results = {
            'dates': data.index,
            'values': data.values,
            'anomaly_scores': anomaly_scores,
            'anomaly_labels': anomaly_labels,
            'statistical_drift': statistical_drift
        }
        
        return results
    
    def plot_results(self, data: pd.Series, results: Dict, title: str = "Data Drift Detection", save_path: str = None):
        """Plot the results with highlighted drift points"""
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Time series with anomalies
        axes[0].plot(results['dates'], results['values'], 
                    label='Data', color='blue', alpha=0.7, linewidth=1)
        
        # Highlight anomalies (outliers)
        anomaly_mask = results['anomaly_labels'] == -1
        if np.any(anomaly_mask):
            anomaly_dates = results['dates'][anomaly_mask]
            anomaly_values = results['values'][anomaly_mask]
            axes[0].scatter(anomaly_dates, anomaly_values, 
                          color='red', s=60, alpha=0.8, 
                          label=f'Anomalies ({np.sum(anomaly_mask)})', 
                          marker='o', edgecolors='darkred')
        
        # Highlight statistical drift periods
        if results['statistical_drift']:
            drift_dates = [d['date'] for d in results['statistical_drift'] if d['drift_detected']]
            if drift_dates:
                drift_values = [data.loc[date] for date in drift_dates]
                axes[0].scatter(drift_dates, drift_values, 
                              color='orange', s=40, alpha=0.6,
                              label=f'Statistical Drift ({len(drift_dates)})', 
                              marker='^')
        
        axes[0].set_title(f'{title} - Time Series with Detected Drift')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        axes[1].plot(results['dates'], results['anomaly_scores'], 
                    color='purple', alpha=0.7, linewidth=1)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, 
                       label='Anomaly Threshold')
        axes[1].fill_between(results['dates'], results['anomaly_scores'], 0, 
                           where=(results['anomaly_labels'] == -1), 
                           color='red', alpha=0.3, label='Anomaly Regions')
        
        axes[1].set_title('Anomaly Scores (Lower = More Anomalous)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Anomaly Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_statistics(self, results: Dict, data_name: str = "Dataset"):
        """Print drift detection statistics"""
        
        total_points = len(results['anomaly_labels'])
        anomalies = np.sum(results['anomaly_labels'] == -1)
        statistical_drift_periods = sum(1 for d in results['statistical_drift'] if d['drift_detected'])
        
        print("\n" + "="*60)
        print(f"DRIFT DETECTION STATISTICS - {data_name}")
        print("="*60)
        print(f"Total data points analyzed: {total_points:,}")
        print(f"Time period: {results['dates'][0].strftime('%Y-%m-%d')} to {results['dates'][-1].strftime('%Y-%m-%d')}")
        print()
        
        print("ANOMALY DETECTION:")
        print(f"Anomalies detected: {anomalies} ({anomalies/total_points*100:.1f}%)")
        print(f"Normal points: {total_points-anomalies} ({(total_points-anomalies)/total_points*100:.1f}%)")
        
        if anomalies > 0:
            anomaly_dates = results['dates'][results['anomaly_labels'] == -1]
            print(f"First anomaly: {anomaly_dates[0].strftime('%Y-%m-%d')}")
            print(f"Last anomaly: {anomaly_dates[-1].strftime('%Y-%m-%d')}")
        
        print()
        print("STATISTICAL DRIFT:")
        print(f"Drift periods detected: {statistical_drift_periods}")
        
        if statistical_drift_periods > 0:
            drift_dates = [d['date'] for d in results['statistical_drift'] if d['drift_detected']]
            print(f"First drift detected: {drift_dates[0].strftime('%Y-%m-%d')}")
            print(f"Last drift detected: {drift_dates[-1].strftime('%Y-%m-%d')}")
        
        print()
        print("DATA SUMMARY:")
        print(f"Data range: {results['values'].min():.1f} to {results['values'].max():.1f}")
        print(f"Mean value: {results['values'].mean():.1f}")
        print(f"Standard deviation: {results['values'].std():.1f}")
        
        # Drift intensity
        if anomalies > 0 or statistical_drift_periods > 0:
            total_drift_events = anomalies + statistical_drift_periods
            print(f"Overall drift intensity: {total_drift_events/total_points*100:.1f}% of data points")
            
            if total_drift_events/total_points > 0.1:
                print("HIGH drift activity detected!")
            elif total_drift_events/total_points > 0.05:
                print("MODERATE drift activity detected")
            else:
                print("LOW drift activity")
        else:
            print("No significant drift detected")
        
        print("="*60)

def main(file_path: str, column_name: str, contamination: float = 0.05, window_size: int = 7):
    """
    Main function to run simplified drift detection
    
    Args:
        file_path: Path to CSV file
        column_name: Name of column to analyze
        contamination: Expected proportion of outliers (0.01-0.2)
        window_size: Rolling window size (3-30)
    """
    
    print(f"Loading data from: {file_path}")
    print(f"Analyzing column: {column_name}")
    print(f"Contamination level: {contamination} (sensitivity to anomalies)")
    print(f"Window size: {window_size}")
    print("-" * 50)
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in data. Available columns: {list(df.columns)}")
        
        data_series = df[column_name]
        print(f"✓ Data loaded successfully: {len(data_series)} data points")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None
    
    # Initialize detector
    detector = LocalDriftDetector(
        contamination=contamination,
        window_size=window_size
    )
    
    # Fit and detect drift
    try:
        detector.fit(data_series)
        results = detector.detect_drift(data_series)
        
        # Create plot title
        dataset_name = file_path.split('/')[-2] if '/' in file_path else 'Dataset'
        plot_title = f"{dataset_name.title()} - {column_name.replace('_', ' ').title()}"
        
        # Plot results
        detector.plot_results(data_series, results, title=plot_title)
        
        # Print statistics
        detector.print_statistics(results, data_name=plot_title)
        
        return detector, results
        
    except Exception as e:
        print(f"Error during drift detection: {e}")
        return None, None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Simple Data Drift Detection Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Required arguments
    parser.add_argument(
        '-f', '--file', 
        required=True, 
        help='Path to CSV file containing time series data'
    )
    
    parser.add_argument(
        '-c', '--column', 
        required=True, 
        help='Name of column to analyze for drift detection'
    )
    
    # Optional parameters
    parser.add_argument(
        '--contamination', 
        type=float, 
        default=0.05,
        help='Expected proportion of outliers (0.01-0.2). Higher = more sensitive. Default: 0.05'
    )
    
    parser.add_argument(
        '--window-size', 
        type=int, 
        default=7,
        help='Rolling window size (3-30). Smaller = more sensitive to short-term changes. Default: 7'
    )
    
    # Preset options
    parser.add_argument(
        '--high-sensitivity', 
        action='store_true',
        help='Use high sensitivity settings (contamination=0.1)'
    )
    
    parser.add_argument(
        '--low-sensitivity', 
        action='store_true',
        help='Use low sensitivity settings (contamination=0.02)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Apply presets
        if args.high_sensitivity:
            args.contamination = 0.1
            print("Applied HIGH SENSITIVITY preset")
        elif args.low_sensitivity:
            args.contamination = 0.02
            print("Applied LOW SENSITIVITY preset")
        
        # Validate parameters
        if not (0.01 <= args.contamination <= 0.2):
            raise ValueError("contamination must be between 0.01 and 0.2")
        if not (3 <= args.window_size <= 30):
            raise ValueError("window_size must be between 3 and 30")
        
        # Run drift detection
        detector, results = main(
            file_path=args.file,
            column_name=args.column,
            contamination=args.contamination,
            window_size=args.window_size
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nUse --help for usage information")