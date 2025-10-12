"""
Drift Detection Comparison Script
Compares Local vs Global drift detection methods for research paper evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
import os
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

# Simple implementations of drift detectors since imports are missing
class LocalDriftDetector:
    def __init__(self, contamination=0.05, window_size=7):
        self.contamination = contamination
        self.window_size = window_size
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        
    def fit(self, data):
        # Fit on historical data (first 70% of data)
        split_point = int(len(data) * 0.7)
        historical_data = data.iloc[:split_point].values.reshape(-1, 1)
        self.isolation_forest.fit(historical_data)
        self.historical_mean = np.mean(historical_data)
        self.historical_std = np.std(historical_data)
        
    def detect_drift(self, data):
        # Detect anomalies
        data_reshaped = data.values.reshape(-1, 1)
        anomaly_labels = self.isolation_forest.predict(data_reshaped)
        
        # Statistical drift detection (simplified)
        statistical_drift = []
        for i in range(self.window_size, len(data)):
            window_data = data.iloc[i-self.window_size:i]
            window_mean = np.mean(window_data)
            z_score = abs(window_mean - self.historical_mean) / self.historical_std
            drift_detected = z_score > 1.5  # 1.5 sigma threshold (more sensitive)
            
            statistical_drift.append({
                'date': data.index[i],
                'drift_detected': drift_detected,
                'z_score': z_score
            })
        
        return {
            'dates': data.index,
            'anomaly_labels': anomaly_labels,
            'statistical_drift': statistical_drift
        }

class GlobalDriftDetector:
    def __init__(self, contamination=0.05, window_size=7, global_weight=0.7):
        self.contamination = contamination
        self.window_size = window_size
        self.global_weight = global_weight
        self.local_detector = LocalDriftDetector(contamination, window_size)
        
    def fit(self, local_data, global_data=None):
        self.local_detector.fit(local_data)
        
        if global_data is not None:
            # Use global data statistics
            self.global_mean = np.mean(global_data)
            self.global_std = np.std(global_data)
        else:
            # Simulate global statistics (for demo purposes)
            self.global_mean = np.mean(local_data) * 1.1  # Slightly different
            self.global_std = np.std(local_data) * 0.9
        
    def detect_drift(self, local_data, global_data=None):
        # Get local results
        local_results = self.local_detector.detect_drift(local_data)
        
        # Global drift detection
        global_drift = []
        for i in range(self.window_size, len(local_data)):
            window_data = local_data.iloc[i-self.window_size:i]
            window_mean = np.mean(window_data)
            
            # Compare against global distribution
            global_z_score = abs(window_mean - self.global_mean) / self.global_std
            global_drift_detected = global_z_score > 1.5  # More sensitive threshold
            
            global_drift.append({
                'date': local_data.index[i],
                'drift_detected': global_drift_detected,
                'global_z_score': global_z_score
            })
        
        return {
            'dates': local_data.index,
            'anomaly_labels': local_results['anomaly_labels'],
            'local_statistical_drift': local_results['statistical_drift'],
            'global_drift': global_drift
        }

class DriftComparisonFramework:
    """
    Comprehensive framework for comparing local vs global drift detection methods
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.results = {}
        
    def load_and_align_data(self, local_file: str, local_column: str, 
                           global_file: str = None) -> Tuple[pd.Series, pd.Series]:
        """
        Load and align local and global datasets
        """
        print(f"Loading data...")
        print(f"Local: {local_file} (column: {local_column})")
        
        # Load local data
        local_df = pd.read_csv(local_file)
        local_df['Date'] = pd.to_datetime(local_df['Date'])
        local_df.set_index('Date', inplace=True)
        
        if local_column not in local_df.columns:
            raise ValueError(f"Column '{local_column}' not found. Available: {list(local_df.columns)}")
        
        local_data = local_df[local_column]
        
        # Load global data if provided
        global_data = None
        if global_file and os.path.exists(global_file):
            print(f"Global: {global_file}")
            global_df = pd.read_csv(global_file)
            global_df['Date'] = pd.to_datetime(global_df['Date'])
            global_df.set_index('Date', inplace=True)
            
            # Find the global column (try Total_<column> first, then first numeric)
            global_column = f"Total_{local_column}"
            if global_column not in global_df.columns:
                numeric_cols = global_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    global_column = numeric_cols[0]
                    print(f"Using '{global_column}' as global column")
                else:
                    raise ValueError("No numeric columns found in global data")
            
            global_data = global_df[global_column]
            
            # Align datasets to common dates
            common_dates = local_data.index.intersection(global_data.index)
            if len(common_dates) == 0:
                print(f"Warning: No common dates found between local and global data")
                print(f"Local data range: {local_data.index.min()} to {local_data.index.max()}")
                print(f"Global data range: {global_data.index.min()} to {global_data.index.max()}")
                # Use local data dates and interpolate global data if needed
                global_data = global_data.reindex(local_data.index, method='nearest')
                common_dates = local_data.index
            
            local_data = local_data.loc[common_dates]
            global_data = global_data.loc[common_dates]
            
            print(f"Aligned to {len(common_dates)} common dates")
        else:
            print(f"No global data provided - local-only comparison")
        
        if len(local_data) == 0:
            raise ValueError("Local data is empty after processing")
        
        print(f"Local data: {len(local_data)} points ({local_data.index[0]} to {local_data.index[-1]})")
        if global_data is not None:
            print(f"Global data: {len(global_data)} points")
        
        return local_data, global_data
    
    def inject_synthetic_drift(self, data: pd.Series, drift_scenarios: List[Dict]) -> Dict:
        """
        Inject controlled drift scenarios for evaluation
        """
        print(f"\nInjecting synthetic drift scenarios...")
        
        synthetic_datasets = {}
        
        for scenario in drift_scenarios:
            scenario_name = scenario['name']
            drift_type = scenario['type']
            start_date = scenario['start_date']
            intensity = scenario.get('intensity', 2.0)
            duration = scenario.get('duration', None)
            
            print(f"Creating scenario: {scenario_name}")
            
            # Create copy of original data
            synthetic_data = data.copy()
            
            # Find start index
            if start_date and start_date in data.index:
                start_idx = data.index.get_loc(start_date)
            else:
                # Find closest date or default to middle
                if start_date:
                    try:
                        # Find closest date
                        closest_idx = np.argmin(np.abs((data.index - pd.Timestamp(start_date)).days))
                        start_idx = closest_idx
                    except:
                        start_idx = len(data) // 2
                else:
                    start_idx = len(data) // 2  # Default to middle
            
            # Determine end index
            if duration:
                end_idx = min(start_idx + duration, len(data))
            else:
                end_idx = len(data)
            
            # Apply drift based on type
            if drift_type == 'none':
                # No changes for control scenario
                pass
            elif drift_type == 'mean_shift':
                # Sudden mean increase
                synthetic_data.iloc[start_idx:end_idx] *= intensity
                
            elif drift_type == 'gradual_shift':
                # Gradual linear increase - use 2*intensity-1 as end multiplier
                # So intensity=1.5 gives range [1.0, 2.0] with average 1.5 (50% increase)
                end_multiplier = 2 * intensity - 1.0
                multiplier = np.linspace(1.0, end_multiplier, end_idx - start_idx)
                synthetic_data.iloc[start_idx:end_idx] *= multiplier
                
            elif drift_type == 'variance_increase':
                # Increased volatility - scale deviations from mean
                window_mean = synthetic_data.iloc[start_idx:end_idx].mean()
                deviations = synthetic_data.iloc[start_idx:end_idx] - window_mean
                # Multiply deviations by intensity to increase variance
                synthetic_data.iloc[start_idx:end_idx] = window_mean + (deviations * intensity)
                
            elif drift_type == 'trend_change':
                # Changed trend direction
                trend = np.linspace(0, intensity * (end_idx - start_idx), end_idx - start_idx)
                synthetic_data.iloc[start_idx:end_idx] += trend
                
            elif drift_type == 'seasonal_disruption':
                # Disrupt seasonal pattern
                seasonal_noise = intensity * np.sin(np.arange(end_idx - start_idx) * 2 * np.pi / 7)
                synthetic_data.iloc[start_idx:end_idx] += seasonal_noise
                
            elif drift_type == 'outlier_burst':
                # Sudden outlier period
                outlier_indices = np.random.choice(
                    range(start_idx, end_idx), 
                    size=min(10, end_idx - start_idx), 
                    replace=False
                )
                synthetic_data.iloc[outlier_indices] *= intensity
            
            # Store synthetic dataset with ground truth
            synthetic_datasets[scenario_name] = {
                'data': synthetic_data,
                'drift_start': start_idx,
                'drift_end': end_idx,
                'drift_dates': data.index[start_idx:end_idx],
                'scenario': scenario
            }
        
        print(f"Created {len(synthetic_datasets)} synthetic scenarios")
        return synthetic_datasets
    
    def run_detection_methods(self, local_data: pd.Series, global_data: pd.Series = None,
                            contamination: float = 0.05, window_size: int = 7,
                            global_weight: float = 0.7) -> Dict:
        """
        Run both local and global drift detection methods
        """
        print(f"\nRunning drift detection methods...")
        print(f"Parameters: contamination={contamination}, window={window_size}, global_weight={global_weight}")
        
        results = {}
        
        # Run Local Drift Detection
        print(f"Running Local Drift Detection...")
        local_detector = LocalDriftDetector(
            contamination=contamination,
            window_size=window_size
        )
        
        try:
            local_detector.fit(local_data)
            local_results = local_detector.detect_drift(local_data)
            results['local'] = {
                'detector': local_detector,
                'results': local_results,
                'success': True,
                'error': None
            }
            print(f"Local detection completed")
        except Exception as e:
            print(f"Local detection failed: {e}")
            results['local'] = {
                'detector': None,
                'results': None,
                'success': False,
                'error': str(e)
            }
        
        # Run Global Drift Detection
        print(f"Running Global Drift Detection...")
        global_detector = GlobalDriftDetector(
            contamination=contamination,
            window_size=window_size,
            global_weight=global_weight
        )
        
        try:
            global_detector.fit(local_data, global_data)
            global_results = global_detector.detect_drift(local_data, global_data)
            results['global'] = {
                'detector': global_detector,
                'results': global_results,
                'success': True,
                'error': None
            }
            print(f"Global detection completed")
        except Exception as e:
            print(f"Global detection failed: {e}")
            results['global'] = {
                'detector': None,
                'results': None,
                'success': False,
                'error': str(e)
            }
        
        return results
    
    def evaluate_detection_performance(self, method_results: Dict, ground_truth: Dict,
                                     tolerance_days: int = 7) -> Dict:
        """
        Evaluate detection performance against ground truth
        """
        if not method_results['success']:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'detection_delay': float('inf'),
                'false_alarm_rate': 1.0,
                'success': False
            }
        
        results = method_results['results']
        
        # Extract detected drift dates
        detected_dates = []
        
        # From anomaly detection
        if 'anomaly_labels' in results:
            anomaly_mask = results['anomaly_labels'] == -1
            if np.any(anomaly_mask):
                detected_dates.extend(results['dates'][anomaly_mask])
        
        # From statistical drift (simplified)
        if 'statistical_drift' in results:
            stat_drift_dates = [d['date'] for d in results['statistical_drift'] if d['drift_detected']]
            detected_dates.extend(stat_drift_dates)
        
        # From local statistical drift (global method)
        if 'local_statistical_drift' in results:
            local_drift_dates = [d['date'] for d in results['local_statistical_drift'] if d['drift_detected']]
            detected_dates.extend(local_drift_dates)
        
        # From global drift (global method)
        if 'global_drift' in results:
            global_drift_dates = [d['date'] for d in results['global_drift'] if d['drift_detected']]
            detected_dates.extend(global_drift_dates)
        
        # Remove duplicates and sort
        detected_dates = sorted(list(set(detected_dates)))
        
        # Ground truth drift period
        true_drift_dates = ground_truth['drift_dates']
        all_dates = results['dates']
        
        # Create binary arrays for evaluation
        try:
            # Convert all dates to comparable pandas Timestamps
            all_dates_ts = []
            for date in all_dates:
                try:
                    all_dates_ts.append(pd.Timestamp(date))
                except:
                    continue
            
            true_drift_dates_ts = []
            for date in true_drift_dates:
                try:
                    true_drift_dates_ts.append(pd.Timestamp(date))
                except:
                    continue
            
            detected_dates_ts = []
            for date in detected_dates:
                try:
                    detected_dates_ts.append(pd.Timestamp(date))
                except:
                    continue
            
            # Create binary arrays
            y_true = np.array([1 if date in true_drift_dates_ts else 0 for date in all_dates_ts])
            y_pred = np.array([1 if date in detected_dates_ts else 0 for date in all_dates_ts])
        except Exception as e:
            print(f"Warning: Could not create evaluation arrays: {e}")
            # Fallback: assume no drift detected
            y_true = np.zeros(len(all_dates))
            y_pred = np.zeros(len(all_dates))
        
        # Calculate metrics
        metrics = {}
        
        if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        else:
            # Handle edge cases
            metrics['precision'] = 1.0 if np.sum(y_pred) == 0 and np.sum(y_true) == 0 else 0.0
            metrics['recall'] = 1.0 if np.sum(y_true) == 0 else 0.0
            metrics['f1_score'] = 1.0 if np.sum(y_pred) == 0 and np.sum(y_true) == 0 else 0.0
            metrics['accuracy'] = 1.0 if np.array_equal(y_true, y_pred) else 0.0
        
        # Detection delay - require sustained detection, not just a single random hit
        if len(true_drift_dates) > 0:
            first_true_drift = pd.Timestamp(min(true_drift_dates))
            
            if detected_dates:
                # Find detections that occurred during drift period
                drift_period_detections = []
                for d in detected_dates:
                    try:
                        d_ts = pd.Timestamp(d)
                        if d_ts >= first_true_drift:
                            drift_period_detections.append(d_ts)
                    except:
                        continue
                
                if drift_period_detections:
                    # Sort detections chronologically
                    drift_period_detections.sort()
                    
                    # Require sustained detection: find first point where we have
                    # detected drift in at least 3-4 detections within the next 14 days
                    window_days = 14  # 2-week window to check for sustained detection
                    min_detections = 5  # Require at least 5 detections
                    sustained_detection_found = False
                    sustained_detection_date = None
                    
                    for i, detection_date in enumerate(drift_period_detections):
                        # Count detections within next window_days
                        future_window_end = detection_date + timedelta(days=window_days)
                        detections_in_window = sum(
                            1 for d in drift_period_detections[i:]
                            if detection_date <= d <= future_window_end
                        )
                        
                        # Require at least min_detections for sustained detection
                        if detections_in_window >= min_detections:
                            sustained_detection_found = True
                            sustained_detection_date = detection_date
                            break
                    
                    if sustained_detection_found:
                        delay_days = (sustained_detection_date - first_true_drift).days
                        metrics['detection_delay'] = max(0, delay_days)
                    else:
                        # Detections exist but not sustained - treat as very late detection
                        # Use the median detection date as a conservative estimate
                        median_detection = drift_period_detections[len(drift_period_detections) // 2]
                        delay_days = (median_detection - first_true_drift).days
                        metrics['detection_delay'] = max(0, delay_days)
                else:
                    # No detections during drift period
                    metrics['detection_delay'] = float('inf')
            else:
                # No detections at all
                metrics['detection_delay'] = float('inf')
        else:
            # No drift in ground truth
            metrics['detection_delay'] = 0.0
        
        # False alarm rate
        try:
            # Convert dates to comparable format
            true_drift_set = set()
            for date in true_drift_dates:
                try:
                    true_drift_set.add(pd.Timestamp(date))
                except:
                    continue
            
            all_dates_set = set()
            for date in all_dates:
                try:
                    all_dates_set.add(pd.Timestamp(date))
                except:
                    continue
            
            detected_dates_set = set()
            for date in detected_dates:
                try:
                    detected_dates_set.add(pd.Timestamp(date))
                except:
                    continue
            
            non_drift_dates = list(all_dates_set - true_drift_set)
            false_alarms = list(detected_dates_set - true_drift_set)
            metrics['false_alarm_rate'] = len(false_alarms) / len(non_drift_dates) if non_drift_dates else 0.0
        except Exception as e:
            print(f"Warning: Could not calculate false alarm rate: {e}")
            metrics['false_alarm_rate'] = 0.0
        
        metrics['success'] = True
        metrics['num_detections'] = len(detected_dates)
        metrics['num_true_drifts'] = len(true_drift_dates)
        
        return metrics
    
    def create_synthetic_data(self, n_points: int = 1000, base_mean: float = 100, 
                             base_std: float = 15) -> pd.Series:
        """
        Create synthetic time series data for testing purposes
        
        Args:
            n_points: Number of data points to generate
            base_mean: Base mean value for the time series
            base_std: Base standard deviation for the time series
            
        Returns:
            pd.Series: Synthetic time series with datetime index
        """
        np.random.seed(self.random_seed)
        
        # Create date range
        start_date = datetime.now() - timedelta(days=n_points)
        dates = [start_date + timedelta(days=i) for i in range(n_points)]
        
        # Generate base time series with some natural variation
        values = np.random.normal(base_mean, base_std, n_points)
        
        # Add some realistic patterns (seasonal and trend components)
        for i in range(n_points):
            # Weekly seasonality
            weekly_pattern = 5 * np.sin(2 * np.pi * i / 7)
            # Small upward trend
            trend = 0.01 * i
            # Small random walk component
            if i > 0:
                values[i] += 0.1 * (values[i-1] - base_mean)
            
            values[i] += weekly_pattern + trend
        
        # Ensure no negative values (relevant for purchase/financial data)
        values = np.maximum(values, 0.1)
        
        return pd.Series(values, index=dates, name='synthetic_data')
    
    def run_comprehensive_comparison(self, local_file: str, local_column: str, 
                                   global_file: str = None, output_dir: str = "comparison_results"):
        """
        Run comprehensive comparison study
        """
        print("COMPREHENSIVE DRIFT DETECTION COMPARISON")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        local_data, global_data = self.load_and_align_data(local_file, local_column, global_file)
        
        # Define 3 sensitivity-based drift scenarios
        drift_scenarios = [
            {
                'name': 'low_sensitivity',
                'type': 'mean_shift',
                'start_date': local_data.index[len(local_data)//2],
                'intensity': 3.0,
                'description': 'High intensity drift - easy to detect (300% mean increase)'
            },
            {
                'name': 'medium_sensitivity',
                'type': 'gradual_shift',
                'start_date': local_data.index[len(local_data)//3],
                'intensity': 1.5,
                'duration': len(local_data)//4,
                'description': 'Medium intensity drift - moderate detection difficulty (gradual 50% increase)'
            },
            {
                'name': 'high_sensitivity',
                'type': 'variance_increase',
                'start_date': local_data.index[2*len(local_data)//3],
                'intensity': 1.3,
                'description': 'Low intensity drift - challenging to detect (30% variance increase)'
            }
        ]
        
        # Run comparison for each scenario
        comparison_results = {}
        
        for scenario in drift_scenarios:
            scenario_name = scenario['name']
            print(f"\nTesting scenario: {scenario_name}")
            print(f"Description: {scenario['description']}")
            
            # Create synthetic drift
            synthetic_datasets = self.inject_synthetic_drift(local_data, [scenario])
            test_data = synthetic_datasets[scenario_name]['data']
            ground_truth = synthetic_datasets[scenario_name]
            
            # Run both detection methods
            detection_results = self.run_detection_methods(
                test_data, global_data,
                contamination=0.05, window_size=7, global_weight=0.7
            )
            
            # Evaluate performance
            local_metrics = self.evaluate_detection_performance(
                detection_results['local'], ground_truth
            )
            global_metrics = self.evaluate_detection_performance(
                detection_results['global'], ground_truth
            )
            
            # Store results
            comparison_results[scenario_name] = {
                'scenario': scenario,
                'ground_truth': ground_truth,
                'local_metrics': local_metrics,
                'global_metrics': global_metrics,
                'detection_results': detection_results
            }
            
            # Print summary
            if local_metrics['success'] and global_metrics['success']:
                print(f"Local F1:  {local_metrics['f1_score']:.3f}")
                print(f"Global F1: {global_metrics['f1_score']:.3f}")
                improvement = ((global_metrics['f1_score'] - local_metrics['f1_score']) / 
                              max(local_metrics['f1_score'], 0.001)) * 100
                print(f"Improvement: {improvement:+.1f}%")
        
        # Generate comprehensive analysis
        self.results = comparison_results
        analysis = self.generate_statistical_analysis()
        
        # Print detailed dataset statistics
        self.print_dataset_statistics(local_data, global_data, comparison_results)
        
        # Save results
        self.save_results(comparison_results, analysis, output_dir)
        self.save_dataset_summaries(local_data, global_data, comparison_results, output_dir)
        
        # Generate visualizations
        self.create_comparison_plots(comparison_results, output_dir)
        self.create_dataset_visualizations(local_data, global_data, comparison_results, output_dir)
        
        print(f"\nComparison complete! Results saved to: {output_dir}")
        
        return comparison_results, analysis
    
    def generate_statistical_analysis(self) -> Dict:
        """
        Generate comprehensive statistical analysis
        """
        print(f"\nGenerating statistical analysis...")
        
        analysis = {}
        
        # Extract metrics for all scenarios
        local_f1_scores = []
        global_f1_scores = []
        local_precision = []
        global_precision = []
        local_recall = []
        global_recall = []
        
        scenario_names = []
        
        for scenario_name, results in self.results.items():
            if (results['local_metrics']['success'] and 
                results['global_metrics']['success']):
                
                scenario_names.append(scenario_name)
                local_f1_scores.append(results['local_metrics']['f1_score'])
                global_f1_scores.append(results['global_metrics']['f1_score'])
                local_precision.append(results['local_metrics']['precision'])
                global_precision.append(results['global_metrics']['precision'])
                local_recall.append(results['local_metrics']['recall'])
                global_recall.append(results['global_metrics']['recall'])
        
        # Overall performance comparison
        analysis['overall'] = {
            'local_mean_f1': np.mean(local_f1_scores),
            'global_mean_f1': np.mean(global_f1_scores),
            'f1_improvement': np.mean(global_f1_scores) - np.mean(local_f1_scores),
            'f1_improvement_pct': ((np.mean(global_f1_scores) - np.mean(local_f1_scores)) / 
                                  max(np.mean(local_f1_scores), 0.001)) * 100,
            'local_std_f1': np.std(local_f1_scores),
            'global_std_f1': np.std(global_f1_scores)
        }
        
        # Statistical significance testing
        if len(local_f1_scores) > 1:
            # Paired t-test
            try:
                t_stat, t_pvalue = stats.ttest_rel(global_f1_scores, local_f1_scores)
                analysis['statistical_tests'] = {
                    't_statistic': t_stat,
                    't_pvalue': t_pvalue,
                    'significant': t_pvalue < 0.05
                }
            except:
                analysis['statistical_tests'] = {
                    't_statistic': None,
                    't_pvalue': None,
                    'significant': False
                }
        
        # Effect size (Cohen's d)
        if len(local_f1_scores) > 0 and len(global_f1_scores) > 0:
            pooled_std = np.sqrt((np.var(local_f1_scores) + np.var(global_f1_scores)) / 2)
            if pooled_std > 0:
                effect_size = (np.mean(global_f1_scores) - np.mean(local_f1_scores)) / pooled_std
            else:
                effect_size = 0.0
            analysis['effect_size'] = effect_size
        
        # Performance by scenario type
        analysis['by_scenario'] = {}
        for i, scenario_name in enumerate(scenario_names):
            analysis['by_scenario'][scenario_name] = {
                'local_f1': local_f1_scores[i],
                'global_f1': global_f1_scores[i],
                'improvement': global_f1_scores[i] - local_f1_scores[i],
                'improvement_pct': ((global_f1_scores[i] - local_f1_scores[i]) / 
                                   max(local_f1_scores[i], 0.001)) * 100
            }
        
        return analysis
    
    def save_results(self, comparison_results: Dict, analysis: Dict, output_dir: str):
        """
        Save detailed results to files
        """
        print(f"Saving results to {output_dir}...")
        
        # Create summary table
        summary_data = []
        for scenario_name, results in comparison_results.items():
            if (results['local_metrics']['success'] and 
                results['global_metrics']['success']):
                
                summary_data.append({
                    'Scenario': scenario_name.replace('_', ' ').title(),
                    'Local_Precision': results['local_metrics']['precision'],
                    'Global_Precision': results['global_metrics']['precision'],
                    'Local_Recall': results['local_metrics']['recall'],
                    'Global_Recall': results['global_metrics']['recall'],
                    'Local_F1': results['local_metrics']['f1_score'],
                    'Global_F1': results['global_metrics']['f1_score'],
                    'F1_Improvement': results['global_metrics']['f1_score'] - results['local_metrics']['f1_score'],
                    'Improvement_Pct': ((results['global_metrics']['f1_score'] - results['local_metrics']['f1_score']) / 
                                       max(results['local_metrics']['f1_score'], 0.001)) * 100,
                    'Local_Detection_Delay': results['local_metrics']['detection_delay'],
                    'Global_Detection_Delay': results['global_metrics']['detection_delay']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/comparison_summary.csv", index=False)
        
        # Save detailed analysis
        with open(f"{output_dir}/statistical_analysis.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif hasattr(obj, 'item'):  # numpy scalars
                    return obj.item()
                return obj
            
            json.dump(convert_numpy(analysis), f, indent=2)
        
        # Create paper-ready table
        paper_table = summary_df[['Scenario', 'Local_F1', 'Global_F1', 'Improvement_Pct']].copy()
        paper_table.columns = ['Drift Type', 'Local F1', 'Global F1', 'Improvement (%)']
        paper_table['Local F1'] = paper_table['Local F1'].round(3)
        paper_table['Global F1'] = paper_table['Global F1'].round(3)
        paper_table['Improvement (%)'] = paper_table['Improvement (%)'].round(1)
        
        paper_table.to_csv(f"{output_dir}/paper_table.csv", index=False)
        
        print(f"Summary table: {output_dir}/comparison_summary.csv")
        print(f"Statistical analysis: {output_dir}/statistical_analysis.json")
        print(f"Paper-ready table: {output_dir}/paper_table.csv")
    
    def create_comparison_plots(self, comparison_results: Dict, output_dir: str):
        """
        Create comprehensive comparison visualizations
        """
        print(f"Creating comparison plots...")
        
        # Extract data for plotting
        scenarios = []
        local_f1 = []
        global_f1 = []
        improvements = []
        
        for scenario_name, results in comparison_results.items():
            if (results['local_metrics']['success'] and 
                results['global_metrics']['success']):
                
                scenarios.append(scenario_name.replace('_', ' ').title())
                local_f1.append(results['local_metrics']['f1_score'])
                global_f1.append(results['global_metrics']['f1_score'])
                improvements.append(((results['global_metrics']['f1_score'] - results['local_metrics']['f1_score']) / 
                                   max(results['local_metrics']['f1_score'], 0.001)) * 100)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: F1 Score Comparison
        x = np.arange(len(scenarios))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, local_f1, width, label='Local Detection', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, global_f1, width, label='Global Detection', alpha=0.8, color='lightcoral')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('F1 Score Comparison by Scenario')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Improvement Percentage
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[0, 1].bar(scenarios, improvements, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Improvement (%)')
        axes[0, 1].set_title('Global Detection Improvement Over Local')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 3: Precision vs Recall
        local_precision = [comparison_results[s.lower().replace(' ', '_')]['local_metrics']['precision'] 
                          for s in scenarios]
        local_recall = [comparison_results[s.lower().replace(' ', '_')]['local_metrics']['recall'] 
                       for s in scenarios]
        global_precision = [comparison_results[s.lower().replace(' ', '_')]['global_metrics']['precision'] 
                           for s in scenarios]
        global_recall = [comparison_results[s.lower().replace(' ', '_')]['global_metrics']['recall'] 
                        for s in scenarios]
        
        axes[1, 0].scatter(local_recall, local_precision, s=100, alpha=0.7, 
                          label='Local Detection', color='skyblue', marker='o')
        axes[1, 0].scatter(global_recall, global_precision, s=100, alpha=0.7, 
                          label='Global Detection', color='lightcoral', marker='s')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, 1.05)
        axes[1, 0].set_ylim(0, 1.05)
        
        # Plot 4: Overall Performance Summary
        metrics = ['Precision', 'Recall', 'F1 Score']
        local_avg = [np.mean(local_precision), np.mean(local_recall), np.mean(local_f1)]
        global_avg = [np.mean(global_precision), np.mean(global_recall), np.mean(global_f1)]
        
        x = np.arange(len(metrics))
        axes[1, 1].bar(x - width/2, local_avg, width, label='Local Detection', alpha=0.8, color='skyblue')
        axes[1, 1].bar(x + width/2, global_avg, width, label='Global Detection', alpha=0.8, color='lightcoral')
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Average Score')
        axes[1, 1].set_title('Overall Performance Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Comparison plots: {output_dir}/comparison_plots.png")

    def create_dataset_visualizations(self, local_data: pd.Series, global_data: pd.Series, 
                                    comparison_results: Dict, output_dir: str):
        """
        Create visualizations of the datasets used in each scenario
        """
        print(f"\nCreating dataset visualizations for each scenario...")
        
        # Create a comprehensive figure showing all scenarios
        n_scenarios = len(comparison_results)
        fig, axes = plt.subplots(n_scenarios, 1, figsize=(12, 4 * n_scenarios))
        
        if n_scenarios == 1:
            axes = [axes]
        
        for idx, (scenario_name, results) in enumerate(comparison_results.items()):
            ax = axes[idx]
            
            # Get the synthetic data for this scenario
            synthetic_data = results['ground_truth']['data']
            drift_start = results['ground_truth']['drift_start']
            drift_end = results['ground_truth']['drift_end']
            scenario_info = results['scenario']
            
            # Plot original local data
            ax.plot(local_data.index, local_data.values, 
                   label='Original Local Data', alpha=0.7, color='blue', linestyle='--')
            
            # Plot global data if available
            if global_data is not None:
                ax.plot(global_data.index, global_data.values, 
                       label='Global Benchmark', alpha=0.7, color='green', linestyle=':')
            
            # Plot synthetic data with drift
            ax.plot(synthetic_data.index, synthetic_data.values, 
                   label='Local Data with Drift', alpha=0.8, color='red', linewidth=2)
            
            # Highlight drift period
            if drift_start < len(synthetic_data) and drift_end <= len(synthetic_data):
                drift_dates = synthetic_data.index[drift_start:drift_end]
                drift_values = synthetic_data.iloc[drift_start:drift_end]
                ax.fill_between(drift_dates, drift_values.min() * 0.9, drift_values.max() * 1.1, 
                              alpha=0.2, color='red', label='Drift Period')
            
            # Formatting
            ax.set_title(f"{scenario_name.replace('_', ' ').title()}: {scenario_info['description']}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dataset_scenarios.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Dataset scenarios plot: {output_dir}/dataset_scenarios.png")

    def print_dataset_statistics(self, local_data: pd.Series, global_data: pd.Series, 
                                comparison_results: Dict):
        """
        Print detailed statistics for local and global datasets in each scenario
        """
        print(f"\nDATASET STATISTICS FOR EACH SCENARIO")
        print("=" * 80)
        
        # Original data statistics
        print(f"\nORIGINAL DATA STATISTICS:")
        print(f"Local Dataset:")
        print(f"  Period: {local_data.index[0]} to {local_data.index[-1]}")
        print(f"  Count: {len(local_data)} points")
        print(f"  Mean: {local_data.mean():.2f}")
        print(f"  Std: {local_data.std():.2f}")
        print(f"  Min: {local_data.min():.2f}")
        print(f"  Max: {local_data.max():.2f}")
        print(f"  Median: {local_data.median():.2f}")
        
        if global_data is not None:
            print(f"\nGlobal Dataset:")
            print(f"  Period: {global_data.index[0]} to {global_data.index[-1]}")
            print(f"  Count: {len(global_data)} points")
            print(f"  Mean: {global_data.mean():.2f}")
            print(f"  Std: {global_data.std():.2f}")
            print(f"  Min: {global_data.min():.2f}")
            print(f"  Max: {global_data.max():.2f}")
            print(f"  Median: {global_data.median():.2f}")
        
        # Scenario-specific statistics
        for scenario_name, results in comparison_results.items():
            print(f"\n{'-' * 60}")
            print(f"SCENARIO: {scenario_name.replace('_', ' ').upper()}")
            print(f"Description: {results['scenario']['description']}")
            print(f"{'-' * 60}")
            
            synthetic_data = results['ground_truth']['data']
            drift_start = results['ground_truth']['drift_start']
            drift_end = results['ground_truth']['drift_end']
            
            # Pre-drift statistics
            pre_drift_data = synthetic_data.iloc[:drift_start]
            drift_data = synthetic_data.iloc[drift_start:drift_end]
            post_drift_data = synthetic_data.iloc[drift_end:] if drift_end < len(synthetic_data) else pd.Series([])
            
            print(f"Local Data with Synthetic Drift:")
            print(f"  Total Count: {len(synthetic_data)} points")
            print(f"  Overall Mean: {synthetic_data.mean():.2f}")
            print(f"  Overall Std: {synthetic_data.std():.2f}")
            
            print(f"\nPre-Drift Period ({len(pre_drift_data)} points):")
            if len(pre_drift_data) > 0:
                print(f"  Period: {pre_drift_data.index[0]} to {pre_drift_data.index[-1]}")
                print(f"  Mean: {pre_drift_data.mean():.2f}")
                print(f"  Std: {pre_drift_data.std():.2f}")
                print(f"  Min: {pre_drift_data.min():.2f}")
                print(f"  Max: {pre_drift_data.max():.2f}")
            
            print(f"\nDrift Period ({len(drift_data)} points):")
            if len(drift_data) > 0:
                print(f"  Period: {drift_data.index[0]} to {drift_data.index[-1]}")
                print(f"  Mean: {drift_data.mean():.2f}")
                print(f"  Std: {drift_data.std():.2f}")
                print(f"  Min: {drift_data.min():.2f}")
                print(f"  Max: {drift_data.max():.2f}")
                
                # Compare to pre-drift
                if len(pre_drift_data) > 0:
                    mean_change = ((drift_data.mean() - pre_drift_data.mean()) / pre_drift_data.mean()) * 100
                    std_change = ((drift_data.std() - pre_drift_data.std()) / pre_drift_data.std()) * 100
                    print(f"  Mean change from pre-drift: {mean_change:+.1f}%")
                    print(f"  Std change from pre-drift: {std_change:+.1f}%")
            
            if len(post_drift_data) > 0:
                print(f"\nPost-Drift Period ({len(post_drift_data)} points):")
                print(f"  Period: {post_drift_data.index[0]} to {post_drift_data.index[-1]}")
                print(f"  Mean: {post_drift_data.mean():.2f}")
                print(f"  Std: {post_drift_data.std():.2f}")
            
            # Performance summary for this scenario
            local_metrics = results['local_metrics']
            global_metrics = results['global_metrics']
            
            print(f"\nDetection Performance:")
            print(f"  Local Method:")
            print(f"    F1-Score: {local_metrics['f1_score']:.3f}")
            print(f"    Precision: {local_metrics['precision']:.3f}")
            print(f"    Recall: {local_metrics['recall']:.3f}")
            print(f"    Detection Delay: {local_metrics['detection_delay']} days")
            
            print(f"  Global Method:")
            print(f"    F1-Score: {global_metrics['f1_score']:.3f}")
            print(f"    Precision: {global_metrics['precision']:.3f}")
            print(f"    Recall: {global_metrics['recall']:.3f}")
            print(f"    Detection Delay: {global_metrics['detection_delay']} days")
            
            improvement = ((global_metrics['f1_score'] - local_metrics['f1_score']) / 
                          max(local_metrics['f1_score'], 0.001)) * 100
            print(f"  Global Improvement: {improvement:+.1f}%")
        
        print(f"\n{'=' * 80}")

    def save_dataset_summaries(self, local_data: pd.Series, global_data: pd.Series, 
                              comparison_results: Dict, output_dir: str):
        """
        Save dataset summaries to CSV files for the report
        """
        print(f"Saving dataset summaries...")
        
        # Create summary for each scenario
        scenario_summaries = []
        
        for scenario_name, results in comparison_results.items():
            synthetic_data = results['ground_truth']['data']
            drift_start = results['ground_truth']['drift_start']
            drift_end = results['ground_truth']['drift_end']
            scenario_info = results['scenario']
            
            # Calculate statistics
            pre_drift = synthetic_data.iloc[:drift_start]
            drift_period = synthetic_data.iloc[drift_start:drift_end]
            
            summary = {
                'Scenario': scenario_name.replace('_', ' ').title(),
                'Description': scenario_info['description'],
                'Drift_Type': scenario_info['type'],
                'Drift_Intensity': scenario_info.get('intensity', 'N/A'),
                'Total_Points': len(synthetic_data),
                'Drift_Start_Date': synthetic_data.index[drift_start],
                'Drift_End_Date': synthetic_data.index[drift_end-1] if drift_end > drift_start else 'N/A',
                'Drift_Duration_Days': drift_end - drift_start,
                'Pre_Drift_Mean': pre_drift.mean() if len(pre_drift) > 0 else 0,
                'Pre_Drift_Std': pre_drift.std() if len(pre_drift) > 0 else 0,
                'Drift_Period_Mean': drift_period.mean() if len(drift_period) > 0 else 0,
                'Drift_Period_Std': drift_period.std() if len(drift_period) > 0 else 0,
                'Mean_Change_Pct': ((drift_period.mean() - pre_drift.mean()) / pre_drift.mean() * 100) if len(pre_drift) > 0 and len(drift_period) > 0 else 0,
                'Std_Change_Pct': ((drift_period.std() - pre_drift.std()) / pre_drift.std() * 100) if len(pre_drift) > 0 and len(drift_period) > 0 else 0,
                'Local_F1': results['local_metrics']['f1_score'],
                'Global_F1': results['global_metrics']['f1_score'],
                'F1_Improvement_Pct': ((results['global_metrics']['f1_score'] - results['local_metrics']['f1_score']) / 
                                      max(results['local_metrics']['f1_score'], 0.001)) * 100
            }
            
            scenario_summaries.append(summary)
        
        # Save to CSV
        summary_df = pd.DataFrame(scenario_summaries)
        summary_df.to_csv(f"{output_dir}/scenario_dataset_summaries.csv", index=False)
        
        # Save individual scenario datasets
        for scenario_name, results in comparison_results.items():
            synthetic_data = results['ground_truth']['data']
            dataset_df = pd.DataFrame({
                'Date': synthetic_data.index,
                'Value': synthetic_data.values,
                'Is_Drift_Period': [1 if i >= results['ground_truth']['drift_start'] and i < results['ground_truth']['drift_end'] else 0 
                                   for i in range(len(synthetic_data))]
            })
            dataset_df.to_csv(f"{output_dir}/dataset_{scenario_name}.csv", index=False)
        
        print(f"Dataset summaries: {output_dir}/scenario_dataset_summaries.csv")
        print(f"Individual datasets: {output_dir}/dataset_*.csv")

    def print_summary_report(self, analysis: Dict):
        """
        Print a comprehensive summary report of the analysis results
        """
        print(f"\n" + "=" * 80)
        print(f"COMPREHENSIVE DRIFT DETECTION COMPARISON SUMMARY")
        print(f"=" * 80)
        
        if 'overall' in analysis:
            overall = analysis['overall']
            print(f"\nOVERALL PERFORMANCE COMPARISON:")
            print(f"  Local Method Average F1:    {overall['local_mean_f1']:.3f}")
            print(f"  Global Method Average F1:   {overall['global_mean_f1']:.3f}")
            print(f"  Absolute Improvement:       {overall['f1_improvement']:+.3f}")
            print(f"  Relative Improvement:       {overall['f1_improvement_pct']:+.1f}%")
            
        if 'statistical_tests' in analysis:
            stats = analysis['statistical_tests']
            print(f"\nSTATISTICAL SIGNIFICANCE:")
            if stats['t_pvalue'] is not None:
                print(f"  T-statistic:                {stats['t_statistic']:.3f}")
                print(f"  P-value:                    {stats['t_pvalue']:.6f}")
                print(f"  Statistically Significant:  {'Yes' if stats['significant'] else 'No'} (p < 0.05)")
            else:
                print(f"  Statistical test not applicable (insufficient data)")
                
        if 'effect_size' in analysis:
            effect_size = analysis['effect_size']
            print(f"  Effect Size (Cohen's d):    {effect_size:.3f}")
            if effect_size < 0.2:
                effect_desc = "Small"
            elif effect_size < 0.5:
                effect_desc = "Small to Medium"
            elif effect_size < 0.8:
                effect_desc = "Medium to Large"
            else:
                effect_desc = "Large"
            print(f"  Effect Size Interpretation: {effect_desc}")
        
        if 'by_scenario' in analysis:
            print(f"\nPERFORMANCE BY SCENARIO:")
            for scenario_name, scenario_results in analysis['by_scenario'].items():
                print(f"\n  {scenario_name.replace('_', ' ').title()}:")
                print(f"    Local F1:       {scenario_results['local_f1']:.3f}")
                print(f"    Global F1:      {scenario_results['global_f1']:.3f}")
                print(f"    Improvement:    {scenario_results['improvement_pct']:+.1f}%")
        
        print(f"\n" + "=" * 80)

def create_test_data():
    """Create test data files for demonstration"""
    framework = DriftComparisonFramework()
    
    # Create local data with a specific date range
    local_data = framework.create_synthetic_data(n_points=1000, base_mean=100, base_std=15)
    local_df = pd.DataFrame({'Date': local_data.index, 'purchase_amount': local_data.values})
    local_df.to_csv('test_local_data.csv', index=False)
    
    # Create global data using the SAME date range as local data
    # Reset the random seed to get different values but same dates
    np.random.seed(43)  # Different seed for different values
    global_values = np.random.normal(105, 12, 1000)  # Slightly different mean and std
    
    # Add similar patterns but slightly different
    for i in range(1000):
        weekly_pattern = 3 * np.sin(2 * np.pi * i / 7)  # Different amplitude
        trend = 0.008 * i  # Slightly different trend
        if i > 0:
            global_values[i] += 0.08 * (global_values[i-1] - 105)
        global_values[i] += weekly_pattern + trend
    
    # Ensure no negative values
    global_values = np.maximum(global_values, 0.1)
    
    # Use the same date index as local data
    global_df = pd.DataFrame({
        'Date': local_data.index, 
        'Total_purchase_amount': global_values
    })
    global_df.to_csv('test_global_data.csv', index=False)
    
    print("Created test_local_data.csv and test_global_data.csv")
    return 'test_local_data.csv', 'test_global_data.csv'

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Drift Detection Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-f', '--file',
                       help='Path to local CSV file containing time series data')
    parser.add_argument('-c', '--column', default='purchase_amount',
                       help='Name of column to analyze for drift detection')
    parser.add_argument('-g', '--global-file',
                       help='Path to global aggregated data CSV file from SMPC protocol')
    parser.add_argument('-o', '--output-dir', default='comparison_results',
                       help='Output directory for results (default: comparison_results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--create-test-data', action='store_true',
                       help='Create synthetic test data and run comparison')
    
    args = parser.parse_args()
    
    try:
        if args.create_test_data or not args.file:
            # Create test data
            local_file, global_file = create_test_data()
            args.file = local_file
            args.global_file = global_file
        
        # Initialize comparison framework
        framework = DriftComparisonFramework(random_seed=args.seed)
        
        # Run comprehensive comparison
        results, analysis = framework.run_comprehensive_comparison(
            local_file=args.file,
            local_column=args.column,
            global_file=args.global_file,
            output_dir=args.output_dir
        )
        
        # Print summary report
        framework.print_summary_report(analysis)
        
        print(f"\nComparison completed successfully!")
        print(f"Results available in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
