
# Weight Calculator Module


import numpy as np
import math
from typing import Dict, Any


class WeightCalculator:

    def __init__(self,
                 lat_weight_factor: float = 2.0,
                 lon_weight_factor: float = 2.0,
                 age_weight_factor: float = 38.0,
                 weight_normalization: float = 0.2):

        # Initialize weight calculator parameters
        self.lat_weight_factor = lat_weight_factor
        self.lon_weight_factor = lon_weight_factor
        self.age_weight_factor = age_weight_factor
        self.weight_normalization = weight_normalization

    def calculate_weights(self,
                          ages: np.ndarray,
                          lats: np.ndarray,
                          lons: np.ndarray) -> np.ndarray:

        # Calculate sample weights
        n_samples = len(ages)

        # Calculate Spatial Kernel
        lat_diff = lats[:, np.newaxis] - lats[np.newaxis, :]
        lon_diff = lons[:, np.newaxis] - lons[np.newaxis, :]
        spatial_kernel = 1.0 / (
        (lat_diff / self.lat_weight_factor) ** 2 +
        (lon_diff / self.lon_weight_factor) ** 2 + 1.0
    )
        
        # Calculate Age Kernel
        age_diff = ages[:, np.newaxis] - ages[np.newaxis, :]
        age_kernel = 1.0 / ((age_diff / self.age_weight_factor) ** 2 + 1.0)

        # Calculate total weight matrix
        weight_matrix = spatial_kernel * age_kernel

        # Calculate the sum of weights for each sample
        weights = np.nansum(weight_matrix, axis=1) * self.weight_normalization

        print(f"Sample weights calculated using optimized method.")
        return weights.reshape(-1, 1)

    def remove_infinite_weights(self,
                                weights: np.ndarray,
                                data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        # Remove data corresponding to infinite weights
        # Find positions of infinite weights
        infinite_indices = np.where(weights == np.inf)[0]

        if len(infinite_indices) > 0:
            print(f"Removed {len(infinite_indices)} samples with infinite weights")

            # Create valid index mask
            valid_mask = np.ones(len(weights), dtype=bool)
            valid_mask[infinite_indices] = False

            # Filter data
            cleaned_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    cleaned_data[key] = value[valid_mask]
                else:
                    cleaned_data[key] = value

            # Filter weights
            cleaned_weights = weights[valid_mask]

            return cleaned_data, cleaned_weights
        else:
            return data, weights

    def get_weight_statistics(self, weights: np.ndarray) -> Dict[str, float]:

        # Get weight statistics
        valid_weights = weights[weights > 0]

        if len(valid_weights) == 0:
            return {
                'mean_weight': 0,
                'std_weight': 0,
                'min_weight': 0,
                'max_weight': 0,
                'total_samples': len(weights),
                'valid_samples': 0
            }

        return {
            'mean_weight': np.mean(valid_weights),
            'std_weight': np.std(valid_weights),
            'min_weight': np.min(valid_weights),
            'max_weight': np.max(valid_weights),
            'total_samples': len(weights),
            'valid_samples': len(valid_weights)
        }

    def save_weights(self,
                     weights: np.ndarray,
                     output_file: str) -> bool:

        # Save weights to file
        try:
            np.save(output_file, weights)
            print(f"Weights saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Failed to save weights: {str(e)}")
            return False

    def load_weights(self,
                     input_file: str) -> np.ndarray:

        # Load weights from file
        try:
            weights = np.load(input_file)
            print(f"Weights loaded from: {input_file}")
            return weights
        except Exception as e:
            print(f"Failed to load weights: {str(e)}")
            return None