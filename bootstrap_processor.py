# Bootstrap Processor Module

import numpy as np
from typing import Dict, Any, List, Tuple
import math


class BootstrapProcessor:

    def __init__(self):
        pass

    def prepare_data(self,
                     data: Dict[str, np.ndarray],
                     weights: np.ndarray) -> Dict[str, np.ndarray]:

        # Prepare required data
        processed_data = data.copy()

        # Ensure no NaN weights
        valid_indices = ~np.isnan(weights)

        if np.any(~valid_indices):
            print(f"Warning: Removed {np.sum(~valid_indices)} samples with NaN weights during preparation.")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    processed_data[key] = value[valid_indices]
            
            weights = weights[valid_indices]

        # Normalize weights
        processed_data['weights'] = weights / np.nansum(weights)

        # Check if the number of samples is too small
        if len(processed_data['age']) < 1:
            raise ValueError("No valid samples remaining after data preparation.")

        return processed_data

    def run_bootstrap(self,
                      processed_data: Dict[str, np.ndarray],
                      iterations: int,
                      time_interval: int,
                      min_age: float = None,
                      max_age: float = None,
                      probability_threshold: float = 0.5) -> np.ndarray:

        # Run Bootstrap resampling analysis

        # Extract data
        ages = processed_data['age']
        age_errors = processed_data['age_error']
        probabilities = processed_data['probability']
        weights = processed_data['weights']

        # If age range is not specified, auto-detect
        if min_age is None:
            min_age = np.min(ages - age_errors)
            min_age = max(0.0, min_age)  # Ensure not negative
        if max_age is None:
            max_age = np.max(ages + age_errors)

        # Robust bin edges and bin count
        if time_interval <= 0:
            raise ValueError("time_interval must be > 0")
        if max_age <= min_age:
            # Expand a tiny epsilon to avoid zero range
            max_age = min_age + time_interval

        # Build bin edges to fully cover [min_age, max_age]
        bin_edges = np.arange(min_age, max_age + time_interval, time_interval)
        n_bins = max(1, len(bin_edges) - 1)

        # Initialize results array
        bootstrap_results = np.full((n_bins, iterations), np.nan)

        print(f"Starting Bootstrap analysis: {iterations} iterations, {n_bins} time bins")

        for i in range(iterations):
            if i % 10 == 0:
                print(f"Iteration progress: {i}/{iterations}")

            # Resample based on weights
            bootstrap_indices = np.random.choice(
                len(ages),
                size=len(ages),
                p=weights.flatten()
            )

            # Vectorized age sampling from normal distribution per picked sample
            sampled_ages = np.random.normal(
                loc=ages[bootstrap_indices],
                scale=age_errors[bootstrap_indices]
            )
            sampled_probabilities = probabilities[bootstrap_indices]

            # Bin assignment: right=True makes bin upper edge inclusive on the right
            bin_ids = np.digitize(sampled_ages, bin_edges, right=True) - 1
            # Clamp bin indices to [0, n_bins-1]
            bin_ids = np.clip(bin_ids, 0, n_bins - 1)

            # Totals per bin
            counts_total = np.bincount(bin_ids, minlength=n_bins).astype(float)
            # Counts above threshold per bin
            mask_above = sampled_probabilities >= probability_threshold
            counts_above = np.bincount(bin_ids[mask_above], minlength=n_bins).astype(float)

            with np.errstate(invalid='ignore', divide='ignore'):
                proportions = (counts_above / counts_total) * 100.0
            proportions[counts_total == 0] = np.nan

            bootstrap_results[:, i] = proportions

        print("Bootstrap analysis completed")
        return bootstrap_results

    def calculate_statistics(self, bootstrap_results: np.ndarray) -> Dict[str, Any]:

        # Calculate mean and standard deviation for each time bin
        mean_values = np.nanmean(bootstrap_results, axis=1)
        std_values = np.nanstd(bootstrap_results, axis=1)

        # Calculate confidence intervals
        confidence_95 = np.percentile(bootstrap_results, [2.5, 97.5], axis=1)
        confidence_90 = np.percentile(bootstrap_results, [5, 95], axis=1)
        confidence_68 = np.percentile(bootstrap_results, [16, 84], axis=1)

        return {
            'mean': mean_values,
            'std': std_values,
            'confidence_95': confidence_95,
            'confidence_90': confidence_90,
            'confidence_68': confidence_68,
            'min_values': np.nanmin(bootstrap_results, axis=1),
            'max_values': np.nanmax(bootstrap_results, axis=1),
            'median_values': np.nanmedian(bootstrap_results, axis=1)
        }

    def run_bootstrap_with_fixed_samples(self,
                                         processed_data: Dict[str, np.ndarray],
                                         iterations: int,
                                         time_interval: int,
                                         min_age: float = None,
                                         max_age: float = None,
                                         probability_threshold: float = 0.5,
                                         fixed_sample_indices: List[int] = None) -> np.ndarray:

        # Extract data
        ages = processed_data['age']
        age_errors = processed_data['age_error']
        probabilities = processed_data['probability']
        weights = processed_data['weights']

        # If age range is not specified, auto-detect
        if min_age is None:
            min_age = np.min(ages - age_errors)
            min_age = max(0.0, min_age)  # Ensure non-negative
        if max_age is None:
            max_age = np.max(ages + age_errors)

        # Robust bin edges and bin count
        if time_interval <= 0:
            raise ValueError("time_interval must be > 0")
        if max_age <= min_age:
            max_age = min_age + time_interval

        bin_edges = np.arange(min_age, max_age + time_interval, time_interval)
        n_bins = max(1, len(bin_edges) - 1)

        # Initialize results array
        bootstrap_results = np.full((n_bins, iterations), np.nan)

        print(f"Starting Bootstrap analysis (including fixed samples): {iterations} iterations")

        for i in range(iterations):
            if i % 10 == 0:
                print(f"Iteration progress: {i}/{iterations}")

            # Process fixed samples
            fixed_ages = []
            fixed_probabilities = []

            if fixed_sample_indices is not None:
                for idx in fixed_sample_indices:
                    # Generate random ages for fixed samples
                    normal_ages = np.random.normal(
                        loc=ages[idx],
                        scale=age_errors[idx],
                        size=100
                    )
                    fixed_age = np.random.choice(normal_ages, size=1)[0]
                    fixed_ages.append(fixed_age)
                    fixed_probabilities.append(probabilities[idx])

            # Resample based on weights (excluding fixed samples)
            if fixed_sample_indices is not None:

                # Create weights excluding fixed samples
                remaining_weights = weights.copy()
                remaining_weights[fixed_sample_indices] = 0
                remaining_weights = remaining_weights / np.nansum(remaining_weights)

                # Resample remaining samples
                remaining_indices = [j for j in range(len(ages)) if j not in fixed_sample_indices]
                bootstrap_indices = np.random.choice(
                    remaining_indices,
                    size=len(remaining_indices),
                    p=remaining_weights[remaining_indices].flatten()
                )
            else:
                bootstrap_indices = np.random.choice(
                    range(len(ages)),
                    size=len(ages),
                    p=weights.flatten()
                )

            # Vectorized generation for resampled part
            sampled_ages = np.random.normal(
                loc=ages[bootstrap_indices],
                scale=age_errors[bootstrap_indices]
            )
            sampled_probabilities = probabilities[bootstrap_indices]

            # Merge fixed samples (if any)
            if fixed_ages:
                all_ages = np.concatenate([np.array(fixed_ages), sampled_ages])
                all_probabilities = np.concatenate([np.array(fixed_probabilities), sampled_probabilities])
            else:
                all_ages = sampled_ages
                all_probabilities = sampled_probabilities

            # Binning
            bin_ids = np.digitize(all_ages, bin_edges, right=True) - 1
            bin_ids = np.clip(bin_ids, 0, n_bins - 1)

            counts_total = np.bincount(bin_ids, minlength=n_bins).astype(float)
            mask_above = all_probabilities >= probability_threshold
            counts_above = np.bincount(bin_ids[mask_above], minlength=n_bins).astype(float)

            with np.errstate(invalid='ignore', divide='ignore'):
                proportions = (counts_above / counts_total) * 100.0
            proportions[counts_total == 0] = np.nan

            bootstrap_results[:, i] = proportions

        print("Bootstrap analysis completed")
        return bootstrap_results

    def save_bootstrap_results(self,
                               bootstrap_results: np.ndarray,
                               output_file: str) -> bool:
        
        # Save bootstrap results
        try:
            np.save(output_file, bootstrap_results)
            print(f"Bootstrap results saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Failed to save bootstrap results: {str(e)}")
            return False

    def load_bootstrap_results(self,
                               input_file: str) -> np.ndarray:

        # Load bootstrap results
        try:
            results = np.load(input_file)
            print(f"Bootstrap results loaded from: {input_file}")
            return results
        except Exception as e:
            print(f"Failed to load bootstrap results: {str(e)}")
            return None