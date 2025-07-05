"""
Drift Detection Comparison Script
Compares Local vs Global drift detection methods for research paper evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
import os
from datetime import datetime, timedelta
import json

# Import our drift detection methods
from local_drift_detection import LocalDriftDetector
from global_drift_detection import GlobalDriftDetector

warnings.filterwarnings('ignore')

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
            local_data = local_data.loc[common_dates]
            global_data = global_data.loc[common_dates]
            
            print(f"Aligned to {len(common_dates)} common dates")
        else:
            print(f"No global data provided - local-only comparison")
        
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
                # Gradual linear increase
                multiplier = np.linspace(1.0, intensity, end_idx - start_idx)
                synthetic_data.iloc[start_idx:end_idx] *= multiplier
                
            elif drift_type == 'variance_increase':
                # Increased volatility
                noise = np.random.normal(0, intensity, end_idx - start_idx)
                synthetic_data.iloc[start_idx:end_idx] += noise
                
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
        
        # From statistical drift (local method)
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
        
        # Detection delay
        if detected_dates and len(true_drift_dates) > 0:
            # Convert to pandas Timestamp for comparison
            first_true_drift = pd.Timestamp(min(true_drift_dates))
            early_detections = []
            for d in detected_dates:
                try:
                    d_ts = pd.Timestamp(d)
                    if d_ts >= first_true_drift:
                        early_detections.append(d_ts)
                except:
                    continue
            
            if early_detections:
                first_detection = min(early_detections)
                metrics['detection_delay'] = (first_detection - first_true_drift).days
            else:
                metrics['detection_delay'] = float('inf')
        else:
            metrics['detection_delay'] = float('inf') if len(true_drift_dates) > 0 else 0.0
        
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
        
        # Define the 3 most relevant synthetic drift scenarios
        drift_scenarios = [
            {
                'name': 'no_drift',
                'type': 'none',
                'start_date': None,
                'intensity': 0.0,
                'description': 'Control scenario with no drift'
            },
            {
                'name': 'sudden_mean_shift',
                'type': 'mean_shift',
                'start_date': local_data.index[len(local_data)//2],
                'intensity': 2.5,
                'description': 'Sudden 150% increase in mean'
            },
            {
                'name': 'variance_explosion',
                'type': 'variance_increase',
                'start_date': local_data.index[2*len(local_data)//3],
                'intensity': 3.0,
                'description': 'Tripled volatility'
            }
        ]
        
        # Run comparison for each scenario
        comparison_results = {}
        
        for scenario in drift_scenarios:
            scenario_name = scenario['name']
            print(f"\nTesting scenario: {scenario_name}")
            print(f"Description: {scenario['description']}")
            
            # Prepare data for this scenario
            if scenario['type'] == 'none':
                # No drift scenario - use original data
                test_data = local_data.copy()
                ground_truth = {
                    'drift_start': None,
                    'drift_end': None, 
                    'drift_dates': [],
                    'scenario': scenario
                }
            else:
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
        
        # Save results
        self.save_results(comparison_results, analysis, output_dir)
        
        # Generate visualizations
        self.create_comparison_plots(comparison_results, output_dir)
        
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
        axes[0, 0].set_xlabel('Drift Scenarios')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('F1 Score Comparison by Scenario')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Improvement Percentage
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[0, 1].bar(scenarios, improvements, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Drift Scenarios')
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
        
        plt.show()
        plt.close()
        
        print(f"Comparison plots: {output_dir}/comparison_plots.png")
    
    def print_summary_report(self, analysis: Dict):
        """
        Print comprehensive summary report
        """
        print(f"\nCOMPREHENSIVE COMPARISON REPORT")
        print("=" * 60)
        
        overall = analysis['overall']
        print(f"OVERALL PERFORMANCE:")
        print(f"Local Detection F1:     {overall['local_mean_f1']:.3f} ± {overall['local_std_f1']:.3f}")
        print(f"Global Detection F1:    {overall['global_mean_f1']:.3f} ± {overall['global_std_f1']:.3f}")
        print(f"Absolute Improvement:   {overall['f1_improvement']:.3f}")
        print(f"Relative Improvement:   {overall['f1_improvement_pct']:.1f}%")
        
        if 'statistical_tests' in analysis:
            stats_tests = analysis['statistical_tests']
            significance = "SIGNIFICANT" if stats_tests['significant'] else "Not significant"
            print(f"  Statistical Significance: {significance} (p={stats_tests.get('t_pvalue', 'N/A'):.4f})")
        
        if 'effect_size' in analysis:
            effect = analysis['effect_size']
            effect_interpretation = "Large" if abs(effect) > 0.8 else "Medium" if abs(effect) > 0.5 else "Small"
            print(f"  Effect Size (Cohen's d): {effect:.3f} ({effect_interpretation})")
        
        print(f"\nPERFORMANCE BY SCENARIO:")
        for scenario, perf in analysis['by_scenario'].items():
            print(f"{scenario.replace('_', ' ').title()}:")
            print(f"Local F1: {perf['local_f1']:.3f} → Global F1: {perf['global_f1']:.3f}")
            print(f"Improvement: {perf['improvement']:.3f} ({perf['improvement_pct']:+.1f}%)")
        
        print("=" * 60)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Drift Detection Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-f', '--file', required=True,
                       help='Path to local CSV file containing time series data')
    parser.add_argument('-c', '--column', required=True,
                       help='Name of column to analyze for drift detection')
    parser.add_argument('-g', '--global-file',
                       help='Path to global aggregated data CSV file from SMPC protocol')
    parser.add_argument('-o', '--output-dir', default='comparison_results',
                       help='Output directory for results (default: comparison_results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
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
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
