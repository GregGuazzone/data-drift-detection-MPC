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

class GlobalDriftDetector:
    """
    Enhanced drift detector that uses global SMPC aggregated data as baseline
    for more accurate drift detection in individual stores/parties
    """
    
    def __init__(self, 
                 contamination: float = 0.05,
                 window_size: int = 7,
                 global_weight: float = 0.7):
        """
        Initialize the Global Drift Detector
        
        Args:
            contamination: Expected proportion of outliers (0.01-0.2)
            window_size: Rolling window size for feature calculation
            global_weight: Weight given to global baseline vs local baseline (0.0-1.0)
        """
        self.contamination = contamination
        self.window_size = window_size
        self.global_weight = global_weight  # How much to weight global vs local baseline
        
        # Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        
        # Baseline statistics
        self.local_baseline_stats = {}
        self.global_baseline_stats = {}
        self.combined_baseline_stats = {}
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

    def _calculate_enhanced_features(self, data: pd.Series, global_data: pd.Series = None) -> pd.DataFrame:
        """
        Calculate enhanced features including global comparisons
        """
        features = pd.DataFrame(index=data.index)
        
        # Original value
        features['value'] = data.values
        
        # Local rolling statistics
        features['local_rolling_mean'] = data.rolling(window=self.window_size, min_periods=1).mean()
        features['local_rolling_std'] = data.rolling(window=self.window_size, min_periods=1).std()
        features['local_rolling_std'] = features['local_rolling_std'].fillna(1.0).replace(0, 1.0)
        
        # Local rate of change
        local_rate_change = data.pct_change().clip(-10, 10)
        features['local_rate_of_change'] = local_rate_change.fillna(0)
        features['local_rate_of_change_abs'] = features['local_rate_of_change'].abs()
        
        # Local trend
        def safe_polyfit(x):
            try:
                if len(x) >= 2:
                    return np.polyfit(range(len(x)), x, 1)[0]
                else:
                    return 0
            except:
                return 0
        
        features['local_trend'] = data.rolling(window=self.window_size, min_periods=2).apply(safe_polyfit).fillna(0)
        
        # Global comparison features (if global data is available)
        if global_data is not None:
            # Align global data with local data by date
            aligned_global = global_data.reindex(data.index, method='nearest').fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Global rolling statistics
            features['global_rolling_mean'] = aligned_global.rolling(window=self.window_size, min_periods=1).mean()
            features['global_rolling_std'] = aligned_global.rolling(window=self.window_size, min_periods=1).std()
            features['global_rolling_std'] = features['global_rolling_std'].fillna(1.0).replace(0, 1.0)
            
            # Deviation from global pattern
            features['deviation_from_global'] = (data - aligned_global) / (aligned_global.abs() + 1)
            features['deviation_from_global_abs'] = features['deviation_from_global'].abs()
            
            # Ratio to global (local/global)
            features['local_to_global_ratio'] = data / (aligned_global + 1)
            
            # Global trend comparison
            global_trend = aligned_global.rolling(window=self.window_size, min_periods=2).apply(safe_polyfit).fillna(0)
            features['trend_deviation'] = features['local_trend'] - global_trend
            
            # Relative volatility (local_std / global_std)
            features['relative_volatility'] = features['local_rolling_std'] / (features['global_rolling_std'] + 1)
            
        # Fill any remaining NaN values and replace infinite values
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Final check - ensure all values are finite
        for col in features.columns:
            features[col] = features[col].where(np.isfinite(features[col]), 0)
        
        return features

    def fit(self, local_data: pd.Series, global_data: pd.Series = None, baseline_split: float = 0.3):
        """
        Fit the model using both local and global baseline data
        
        Args:
            local_data: Local time series data (e.g., individual store sales)
            global_data: Global aggregated data from SMPC protocol
            baseline_split: Fraction of data to use for baseline (0.2-0.5)
        """
        # Clean input data
        local_data = self._clean_data(local_data)
        if global_data is not None:
            global_data = self._clean_data(global_data)
        
        # Use first portion as baseline
        split_idx = max(20, int(len(local_data) * baseline_split))
        local_baseline = local_data.iloc[:split_idx]
        
        # Store local baseline statistics
        self.local_baseline_stats = {
            'mean': local_baseline.mean(),
            'std': max(local_baseline.std(), 1.0),
            'median': local_baseline.median(),
            'data': local_baseline.values
        }
        
        # Store global baseline statistics if available
        if global_data is not None:
            global_baseline = global_data.iloc[:split_idx] if len(global_data) >= split_idx else global_data
            self.global_baseline_stats = {
                'mean': global_baseline.mean(),
                'std': max(global_baseline.std(), 1.0),
                'median': global_baseline.median(),
                'data': global_baseline.values
            }
            
            # Create combined baseline using weighted average
            combined_mean = (self.global_weight * self.global_baseline_stats['mean'] + 
                           (1 - self.global_weight) * self.local_baseline_stats['mean'])
            combined_std = (self.global_weight * self.global_baseline_stats['std'] + 
                          (1 - self.global_weight) * self.local_baseline_stats['std'])
            combined_median = (self.global_weight * self.global_baseline_stats['median'] + 
                             (1 - self.global_weight) * self.local_baseline_stats['median'])
            
            self.combined_baseline_stats = {
                'mean': combined_mean,
                'std': combined_std,
                'median': combined_median
            }
        else:
            # Use only local baseline if no global data
            self.global_baseline_stats = None
            self.combined_baseline_stats = self.local_baseline_stats.copy()
        
        # Calculate enhanced features for baseline
        baseline_features = self._calculate_enhanced_features(local_baseline, 
                                                            global_data.iloc[:split_idx] if global_data is not None else None)
        
        # Clean features for sklearn
        baseline_features = self._clean_features_for_sklearn(baseline_features)
        
        try:
            # Fit scaler and isolation forest
            baseline_scaled = self.scaler.fit_transform(baseline_features)
            
            # Final check for finite values
            if not np.all(np.isfinite(baseline_scaled)):
                print("Warning: Non-finite values detected, applying final cleaning...")
                baseline_scaled = np.nan_to_num(baseline_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
            
            self.isolation_forest.fit(baseline_scaled)
            self.is_fitted = True
            
            print(f"Global Drift Detector fitted successfully")
            print(f"Local baseline: {len(local_baseline)} samples ({local_baseline.index[0].strftime('%Y-%m-%d')} to {local_baseline.index[-1].strftime('%Y-%m-%d')})")
            print(f"Local stats: mean={self.local_baseline_stats['mean']:.1f}, std={self.local_baseline_stats['std']:.1f}")
            
            if global_data is not None:
                print(f"Global baseline: {len(global_baseline)} samples")
                print(f"Global stats: mean={self.global_baseline_stats['mean']:.1f}, std={self.global_baseline_stats['std']:.1f}")
                print(f"Combined stats: mean={self.combined_baseline_stats['mean']:.1f}, std={self.combined_baseline_stats['std']:.1f}")
                print(f"Global weight: {self.global_weight:.1f}")
            else:
                print("No global data provided - using local baseline only")
            
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

    def detect_drift(self, local_data: pd.Series, global_data: pd.Series = None) -> Dict:
        """
        Detect drift using enhanced global+local analysis
        
        Args:
            local_data: Local time series data to analyze
            global_data: Global aggregated data for comparison
            
        Returns:
            Dictionary with comprehensive drift detection results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting drift")
        
        # Clean input data
        local_data = self._clean_data(local_data)
        if global_data is not None:
            global_data = self._clean_data(global_data)
        
        # Calculate enhanced features
        features = self._calculate_enhanced_features(local_data, global_data)
        features = self._clean_features_for_sklearn(features)
        
        # Scale features and detect anomalies
        features_scaled = self.scaler.transform(features)
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        anomaly_labels = self.isolation_forest.predict(features_scaled)
        
        # Enhanced statistical drift detection using global baseline
        statistical_drift = []
        global_drift = []
        window_size = min(60, max(30, len(local_data) // 8))
        
        # Check every 7 days for stability
        for i in range(window_size, len(local_data), 7):
            window_data = local_data.iloc[i-window_size:i]
            
            # Local drift detection (against local baseline)
            local_drift_detected = False
            local_p_value = 1.0
            
            try:
                # Statistical tests against local baseline
                _, local_ks_p = stats.ks_2samp(self.local_baseline_stats['data'], window_data.values)
                _, local_ttest_p = stats.ttest_ind(self.local_baseline_stats['data'], window_data.values)
                
                local_drift_criteria = [
                    local_ks_p < 0.01,
                    local_ttest_p < 0.01
                ]
                local_drift_detected = any(local_drift_criteria)
                local_p_value = min(local_ks_p, local_ttest_p)
                
            except:
                pass
            
            statistical_drift.append({
                'date': local_data.index[i],
                'drift_detected': local_drift_detected,
                'p_value': local_p_value
            })
            
            # Global drift detection (if global data available)
            if global_data is not None and self.global_baseline_stats is not None:
                global_drift_detected = False
                global_p_value = 1.0
                
                try:
                    # Get corresponding global window
                    global_window = global_data.iloc[i-window_size:i] if i <= len(global_data) else global_data.iloc[-window_size:]
                    
                    # Compare local window against global baseline
                    _, global_ks_p = stats.ks_2samp(self.global_baseline_stats['data'], window_data.values)
                    _, global_ttest_p = stats.ttest_ind(self.global_baseline_stats['data'], window_data.values)
                    
                    # Compare local window against current global pattern
                    _, pattern_ks_p = stats.ks_2samp(global_window.values, window_data.values)
                    
                    # Deviation from global pattern
                    deviation_threshold = 2.0 * self.global_baseline_stats['std']
                    mean_deviation = abs(window_data.mean() - global_window.mean())
                    
                    global_drift_criteria = [
                        global_ks_p < 0.005,  # Stricter for global comparison
                        global_ttest_p < 0.005,
                        pattern_ks_p < 0.01,
                        mean_deviation > deviation_threshold
                    ]
                    
                    global_drift_detected = any(global_drift_criteria)
                    global_p_value = min(global_ks_p, global_ttest_p, pattern_ks_p)
                    
                except:
                    pass
                
                global_drift.append({
                    'date': local_data.index[i],
                    'drift_detected': global_drift_detected,
                    'p_value': global_p_value,
                    'deviation_from_global': mean_deviation if 'mean_deviation' in locals() else 0
                })
        
        # Combine results
        results = {
            'dates': local_data.index,
            'local_values': local_data.values,
            'global_values': global_data.values if global_data is not None else None,
            'anomaly_scores': anomaly_scores,
            'anomaly_labels': anomaly_labels,
            'local_statistical_drift': statistical_drift,
            'global_drift': global_drift if global_data is not None else [],
            'has_global_data': global_data is not None
        }
        
        return results
    
    def plot_results(self, local_data: pd.Series, results: Dict, title: str = "Global Data Drift Detection", save_path: str = None):
        """Plot comprehensive drift detection results"""
        
        n_plots = 3 if results['has_global_data'] else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 12))
        
        # Plot 1: Local time series with anomalies and local drift
        axes[0].plot(results['dates'], results['local_values'], 
                    label='Local Data', color='blue', alpha=0.7, linewidth=1)
        
        # Highlight anomalies
        anomaly_mask = results['anomaly_labels'] == -1
        if np.any(anomaly_mask):
            anomaly_dates = results['dates'][anomaly_mask]
            anomaly_values = results['local_values'][anomaly_mask]
            axes[0].scatter(anomaly_dates, anomaly_values, 
                          color='red', s=60, alpha=0.8, 
                          label=f'Anomalies ({np.sum(anomaly_mask)})', 
                          marker='o', edgecolors='darkred')
        
        # Highlight local statistical drift
        if results['local_statistical_drift']:
            local_drift_dates = [d['date'] for d in results['local_statistical_drift'] if d['drift_detected']]
            if local_drift_dates:
                local_drift_values = [local_data.loc[date] for date in local_drift_dates]
                axes[0].scatter(local_drift_dates, local_drift_values, 
                              color='orange', s=40, alpha=0.6,
                              label=f'Local Drift ({len(local_drift_dates)})', 
                              marker='^')
        
        axes[0].set_title(f'{title} - Local Data with Anomalies and Local Drift')
        axes[0].set_ylabel('Local Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Global comparison (if available)
        if results['has_global_data']:
            axes[1].plot(results['dates'], results['local_values'], 
                        label='Local Data', color='blue', alpha=0.7, linewidth=1)
            axes[1].plot(results['dates'], results['global_values'], 
                        label='Global Baseline', color='green', alpha=0.7, linewidth=2)
            
            # Highlight global drift
            if results['global_drift']:
                global_drift_dates = [d['date'] for d in results['global_drift'] if d['drift_detected']]
                if global_drift_dates:
                    global_drift_values = [local_data.loc[date] for date in global_drift_dates]
                    axes[1].scatter(global_drift_dates, global_drift_values, 
                                  color='purple', s=50, alpha=0.8,
                                  label=f'Global Drift ({len(global_drift_dates)})', 
                                  marker='s')
            
            axes[1].set_title('Local vs Global Comparison with Global Drift Detection')
            axes[1].set_ylabel('Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plot_idx = 2
        else:
            plot_idx = 1
        
        # Plot: Anomaly scores
        axes[plot_idx].plot(results['dates'], results['anomaly_scores'], 
                           color='purple', alpha=0.7, linewidth=1)
        axes[plot_idx].axhline(y=0, color='red', linestyle='--', alpha=0.7, 
                              label='Anomaly Threshold')
        axes[plot_idx].fill_between(results['dates'], results['anomaly_scores'], 0, 
                                   where=(results['anomaly_labels'] == -1), 
                                   color='red', alpha=0.3, label='Anomaly Regions')
        
        axes[plot_idx].set_title('Anomaly Scores (Lower = More Anomalous)')
        axes[plot_idx].set_xlabel('Date')
        axes[plot_idx].set_ylabel('Anomaly Score')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_statistics(self, results: Dict, data_name: str = "Dataset"):
        """Print comprehensive drift detection statistics"""
        
        total_points = len(results['anomaly_labels'])
        anomalies = np.sum(results['anomaly_labels'] == -1)
        local_drift_periods = sum(1 for d in results['local_statistical_drift'] if d['drift_detected'])
        global_drift_periods = sum(1 for d in results['global_drift'] if d['drift_detected']) if results['has_global_data'] else 0
        
        print("\n" + "="*70)
        print(f"GLOBAL DRIFT DETECTION STATISTICS - {data_name}")
        print("="*70)
        print(f"Total data points analyzed: {total_points:,}")
        print(f"Time period: {results['dates'][0].strftime('%Y-%m-%d')} to {results['dates'][-1].strftime('%Y-%m-%d')}")
        print(f"Global baseline available: {'Yes' if results['has_global_data'] else 'No'}")
        print()
        
        print("ANOMALY DETECTION:")
        print(f"Anomalies detected: {anomalies} ({anomalies/total_points*100:.1f}%)")
        
        print()
        print("LOCAL DRIFT DETECTION:")
        print(f"Local drift periods: {local_drift_periods}")
        
        if results['has_global_data']:
            print()
            print("GLOBAL DRIFT DETECTION:")
            print(f"Global drift periods: {global_drift_periods}")
            print(f"Periods drifting from global pattern: {global_drift_periods}")
        
        print()
        print("DATA SUMMARY:")
        print(f"Local data range: {results['local_values'].min():.1f} to {results['local_values'].max():.1f}")
        print(f"Local mean: {results['local_values'].mean():.1f}")
        print(f"Local std: {results['local_values'].std():.1f}")
        
        if results['has_global_data']:
            print(f"Global data range: {results['global_values'].min():.1f} to {results['global_values'].max():.1f}")
            print(f"Global mean: {results['global_values'].mean():.1f}")
            print(f"Global std: {results['global_values'].std():.1f}")
        
        print()
        print("DRIFT SUMMARY:")
        total_drift_events = anomalies + local_drift_periods + global_drift_periods
        drift_intensity = total_drift_events / total_points
        
        print(f"Total drift events: {total_drift_events}")
        print(f"Drift intensity: {drift_intensity*100:.1f}% of data points")
        
        if drift_intensity > 0.15:
            print("IGH drift activity detected!")
        elif drift_intensity > 0.08:
            print("MODERATE drift activity detected")
        elif drift_intensity > 0.03:
            print("LOW drift activity detected")
        else:
            print("Minimal drift detected - data appears stable")
        
        if results['has_global_data']:
            print()
            print("GLOBAL CONTEXT:")
            if global_drift_periods > 0:
                print(f"This store shows deviation from global patterns")
                print(f"Consider investigating local factors affecting this location")
            else:
                print(f"This store follows global patterns closely")
                print(f"Local behavior is consistent with network-wide trends")
        
        print("="*70)


def main(local_file: str, local_column: str, global_file: str = None, 
         contamination: float = 0.05, window_size: int = 7, 
         global_weight: float = 0.7):
    """
    Main function to run global drift detection
    
    Args:
        local_file: Path to local data CSV file
        local_column: Column name in local data to analyze
        global_file: Path to global aggregated data CSV file (optional)
        contamination: Expected proportion of outliers (0.01-0.2)
        window_size: Rolling window size (3-30)
        global_weight: Weight for global vs local baseline (0.0-1.0)
    """
    
    print(f"GLOBAL DRIFT DETECTION")
    print(f"Local data: {local_file}")
    print(f"Local column: {local_column}")
    print(f"Global data: {global_file if global_file else 'None (local-only mode)'}")
    print(f"Parameters: contamination={contamination}, window={window_size}, global_weight={global_weight}")
    print("-" * 60)
    
    # Load local data
    try:
        local_df = pd.read_csv(local_file)
        local_df['Date'] = pd.to_datetime(local_df['Date'])
        local_df.set_index('Date', inplace=True)
        
        if local_column not in local_df.columns:
            raise ValueError(f"Column '{local_column}' not found in local data. Available: {list(local_df.columns)}")
        
        local_series = local_df[local_column]
        print(f"Local data loaded: {len(local_series)} points")
        
    except Exception as e:
        print(f"Error loading local data: {e}")
        return None, None
    
    # Load global data if provided
    global_series = None
    if global_file:
        try:
            global_df = pd.read_csv(global_file)
            global_df['Date'] = pd.to_datetime(global_df['Date'])
            global_df.set_index('Date', inplace=True)
            
            # Try to find the corresponding global column
            global_column_name = f"Total_{local_column}"
            if global_column_name not in global_df.columns:
                # Fallback to first numeric column
                numeric_cols = global_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    global_column_name = numeric_cols[0]
                    print(f"Column 'Total_{local_column}' not found, using '{global_column_name}'")
                else:
                    raise ValueError("No numeric columns found in global data")
            
            global_series = global_df[global_column_name]
            print(f"✓ Global data loaded: {len(global_series)} points from column '{global_column_name}'")
            
        except Exception as e:
            print(f"Warning: Could not load global data: {e}")
            print("Proceeding with local-only analysis...")
    
    # Initialize detector
    detector = GlobalDriftDetector(
        contamination=contamination,
        window_size=window_size,
        global_weight=global_weight
    )
    
    # Fit and detect drift
    try:
        detector.fit(local_series, global_series)
        results = detector.detect_drift(local_series, global_series)
        
        # Create output files
        local_name = local_file.split('/')[-2] if '/' in local_file else 'Dataset'
        plot_title = f"{local_name.title()} - {local_column.replace('_', ' ').title()}"
        plot_filename = f"global_drift_analysis_{local_name}_{local_column}.png"
        
        # Plot and analyze results
        detector.plot_results(local_series, results, title=plot_title, save_path=plot_filename)
        detector.print_statistics(results, data_name=plot_title)
        
        return detector, results
        
    except Exception as e:
        print(f"Error during drift detection: {e}")
        return None, None


def parse_arguments():
    """Parse command line arguments for global drift detection"""
    parser = argparse.ArgumentParser(
        description='Global Data Drift Detection Tool - Enhanced with SMPC Global Baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '-f', '--file', 
        required=True, 
        help='Path to local CSV file containing time series data'
    )
    
    parser.add_argument(
        '-c', '--column', 
        required=True, 
        help='Name of column to analyze for drift detection'
    )
    
    # Optional global data
    parser.add_argument(
        '-g', '--global-file', 
        help='Path to global aggregated data CSV file from SMPC protocol'
    )
    
    # Detection parameters
    parser.add_argument(
        '--contamination', 
        type=float, 
        default=0.05,
        help='Expected proportion of outliers (0.01-0.2). Default: 0.05'
    )
    
    parser.add_argument(
        '--window-size', 
        type=int, 
        default=7,
        help='Rolling window size (3-30). Default: 7'
    )
    
    parser.add_argument(
        '--global-weight', 
        type=float, 
        default=0.7,
        help='Weight for global vs local baseline (0.0-1.0). Higher = more global influence. Default: 0.7'
    )
    
    # Preset options
    parser.add_argument(
        '--high-sensitivity', 
        action='store_true',
        help='Use high sensitivity settings (contamination=0.1, global_weight=0.8)'
    )
    
    parser.add_argument(
        '--low-sensitivity', 
        action='store_true',
        help='Use low sensitivity settings (contamination=0.02, global_weight=0.5)'
    )
    
    parser.add_argument(
        '--local-only', 
        action='store_true',
        help='Use local baseline only (global_weight=0.0)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Apply presets
        if args.high_sensitivity:
            args.contamination = 0.1
            args.global_weight = 0.8
            print("Applied HIGH SENSITIVITY preset")
        elif args.low_sensitivity:
            args.contamination = 0.02
            args.global_weight = 0.5
            print("Applied LOW SENSITIVITY preset")
        elif args.local_only:
            args.global_weight = 0.0
            print("Applied LOCAL ONLY preset")
        
        # Validate parameters
        if not (0.01 <= args.contamination <= 0.2):
            raise ValueError("contamination must be between 0.01 and 0.2")
        if not (3 <= args.window_size <= 30):
            raise ValueError("window_size must be between 3 and 30")
        if not (0.0 <= args.global_weight <= 1.0):
            raise ValueError("global_weight must be between 0.0 and 1.0")
        
        # Run global drift detection
        detector, results = main(
            local_file=args.file,
            local_column=args.column,
            global_file=args.global_file,
            contamination=args.contamination,
            window_size=args.window_size,
            global_weight=args.global_weight
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nUse --help for usage information")
