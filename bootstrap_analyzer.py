# Bootstrap Analyzer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
import warnings

from .data_loader import DataLoader
from .weight_calculator import WeightCalculator
from .bootstrap_processor import BootstrapProcessor
from .plotter import Plotter


class BootstrapAnalyzer:

    def __init__(self,
                 time_interval: int = 100,
                 iterations: int = 100,
                 random_seed: int = 2025,
                 lat_weight_factor: float = 2.0,
                 lon_weight_factor: float = 2.0,
                 age_weight_factor: float = 38.0,
                 weight_normalization: float = 0.2,
                 ):

        # Initialize Bootstrap Analyzer
        self.time_interval = time_interval
        self.iterations = iterations
        self.random_seed = random_seed
        self.lat_weight_factor = lat_weight_factor
        self.lon_weight_factor = lon_weight_factor
        self.age_weight_factor = age_weight_factor
        self.weight_normalization = weight_normalization

        # Set random seed
        np.random.seed(random_seed)

        # Initialize components
        self.data_loader = DataLoader()
        self.weight_calculator = WeightCalculator(
            lat_weight_factor=lat_weight_factor,
            lon_weight_factor=lon_weight_factor,
            age_weight_factor=age_weight_factor,
            weight_normalization=weight_normalization
        )
        self.bootstrap_processor = BootstrapProcessor()
        self.plotter = Plotter()

        # Data storage
        self.data = None
        self.weights = None
        self.bootstrap_results = None
        self.processed_data = None

        # Dynamic age range detection
        self.detected_min_age = None
        self.detected_max_age = None

    def _detect_age_range(self) -> Tuple[float, float]:

        # Automatically detect the age range of the data
        if self.data is None:
            return 0.0, 3800.0
            
        ages = self.data['age']
        age_errors = self.data['age_error']
        
        # Consider age error, calculate actual min and max age range
        min_age_with_error = np.min(ages - age_errors)
        max_age_with_error = np.max(ages + age_errors)
        
        # Add some buffer space (5%)
        buffer = (max_age_with_error - min_age_with_error) * 0.05
        detected_min_age = max(0.0, min_age_with_error - buffer)
        detected_max_age = max_age_with_error + buffer
        
        return detected_min_age, detected_max_age

    def suggest_time_interval(self, target_bins: int = 15) -> int:

        # Suggest appropriate time intervals based on data range
        if self.data is None:
            return self.time_interval
            
        detected_min_age, detected_max_age = self._detect_age_range()
        age_range = detected_max_age - detected_min_age
        
        if age_range <= 0:
            return self.time_interval
            
        # Calculate suggested time intervals
        suggested_interval = max(1, int(age_range / target_bins))

        return min(suggested_interval, 500)

    def auto_adjust_time_interval(self, target_bins: int = 15) -> bool:

        # Auto-adjust time intervals
        if self.data is None:
            print("error: Please load the data first")
            return False
            
        old_interval = self.time_interval
        new_interval = self.suggest_time_interval(target_bins)
        
        if new_interval != old_interval:
            self.time_interval = new_interval
            print(f" Time interval has been adjusted from {old_interval} Ma to {new_interval} Ma")
            print(f"Approximately {target_bins} time intervals will be generated")
            return True
        else:
            print(f"Current time interval {old_interval} Ma is already appropriate")
            return False

    # Load Excel data
    def load_data(self, file_path: str, **kwargs) -> bool:

        try:
            self.data = self.data_loader.load_excel_data(file_path, **kwargs)

            # Automatically detect age range
            self.detected_min_age, self.detected_max_age = self._detect_age_range()

            print(f"Successfully loaded data, total {len(self.data)} samples")
            print(f"Detected age range: {self.detected_min_age:.1f} - {self.detected_max_age:.1f} Ma")
            return True
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            return False

    # Calculate weights
    def calculate_weights(self) -> bool:

        if self.data is None:
            print("error: Please load the data first")
            return False

        try:
            self.weights = self.weight_calculator.calculate_weights(
                ages=self.data['age'],
                lats=self.data['lat'],
                lons=self.data['lon']
            )
            print("Sample weights calculated successfully.")
            return True
        except Exception as e:
            print(f"Failed to calculate weights: {str(e)}")
            return False

    # Run Bootstrap analysis
    def run_bootstrap_analysis(self,
                               min_age: Optional[float] = None,
                               max_age: Optional[float] = None,
                               probability_threshold: float = 0.5) -> bool:

        if self.data is None or self.weights is None:
            print("error: Please load the data and calculate the weights first")
            return False

        # Use detected age range or user-specified range
        if min_age is None:
            min_age = self.detected_min_age
        if max_age is None:
            max_age = self.detected_max_age

        # Process data
        try:
            self.weights, self.data = self.weight_calculator.remove_infinite_weights(
                weights=self.weights,
                data=self.data
            )

            if len(self.data['age']) == 0:
                print("Error: No valid samples remain after removing infinite weights.")
                return False
            
            # Prepare data 
            self.processed_data = self.bootstrap_processor.prepare_data(
                data=self.data,
                weights=self.weights
            )

            # Determine age range and bin edges
            data_ages = self.processed_data['age']

            if min_age is None:
                min_age = np.floor(np.min(data_ages) / self.time_interval) * self.time_interval
            
            if max_age is None:
                max_age = np.ceil(np.max(data_ages) / self.time_interval) * self.time_interval

            # Store final age range
            self.final_min_age = min_age
            self.final_max_age = max_age
            
            bin_edges = np.arange(min_age, max_age + self.time_interval, self.time_interval)

            # Run Bootstrap analysis
            self.bootstrap_results = self.bootstrap_processor.run_bootstrap(
                processed_data=self.processed_data,
                iterations=self.iterations,
                time_interval=self.time_interval,
                min_age=min_age,
                max_age=max_age,
                probability_threshold=probability_threshold
            )

            print(f"Bootstrap analysis completed, {self.iterations} iterations")
            print(f"Using age range: {min_age:.1f} - {max_age:.1f} Ma")
            return True
        except Exception as e:
            print(f"Bootstrap analysis failed: {str(e)}")
            return False

    def plot_results(self,
                     output_file: str = 'bootstrap_analysis.pdf',
                     title: str = 'Bootstrap Analysis Results',
                     xlabel: str = 'Age (Ma)',
                     ylabel: str = 'Probability Ratio (%)',
                     confidence_level: float = 0.95,
                     min_age: Optional[float] = None,
                     max_age: Optional[float] = None) -> bool:

        # Plot analysis results
        if self.bootstrap_results is None:
            print("Error: Bootstrap results not available. Cannot plot.")
            return False
        
        if min_age is None:
            if self.final_min_age is None:
             print("Error: final_min_age is not set. Run analysis first.")
             return False
        min_age = self.final_min_age

        if max_age is None:
            if self.final_max_age is None:
             print("Error: final_max_age is not set. Run analysis first.")
             return False
        max_age = self.final_max_age

        try:
            self.plotter.plot_bootstrap_results(
                bootstrap_results=self.bootstrap_results,
                time_interval=self.time_interval,
                output_file=output_file,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                confidence_level=confidence_level,
                min_age=min_age,
                max_age=max_age
            )
            print(f"Chart saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Plotting error: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:

        # Get statistics
        if self.bootstrap_results is None:
            print("error: Please run the Bootstrap analysis first")
            return {}

        return self.bootstrap_processor.calculate_statistics(self.bootstrap_results)

    
    def get_data_summary(self) -> Dict[str, Any]:

        # Get data summary
        if self.data is None:
            return {}
            
        summary = self.data_loader.get_data_summary(self.data)
        summary.update({
            'detected_min_age': self.detected_min_age,
            'detected_max_age': self.detected_max_age,
            'time_interval': self.time_interval,
            'iterations': self.iterations
        })
        return summary

    def run_complete_analysis(self,
                              file_path: str,
                              age_column: str,
                              age_max_column: str,
                              probability_column: str,
                              lat_column: str,
                              lon_column: str,
                              min_age: Optional[float] = None,
                              max_age: Optional[float] = None,
                              probability_threshold: float = 0.5,
                              output_file: str = 'bootstrap_analysis.pdf',
                              **kwargs) -> bool:

        # Run the complete analysis process
        try:
            # 1. Load data
            if not self.load_data(
                    file_path=file_path,
                    age_column=age_column,
                    age_max_column=age_max_column,
                    probability_column=probability_column,
                    lat_column=lat_column,
                    lon_column=lon_column,
                    **kwargs
            ):
                return False

            # 2. Calculate weights
            if not self.calculate_weights():
                return False

            # 3. Run Bootstrap analysis
            if not self.run_bootstrap_analysis(
                    min_age=min_age,
                    max_age=max_age,
                    probability_threshold=probability_threshold
            ):
                return False

            # 4. Plot results
            if not self.plot_results(
                    output_file=output_file,
                    min_age=min_age,
                    max_age=max_age
            ):
                return False

            print("The complete analysis process has been finished")
            return True

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return False