# Data Loader Module

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os


class DataLoader:

    def __init__(self):
        pass

    def load_excel_data(self,
                        file_path: str,
                        age_column: str,
                        age_max_column: str,
                        probability_column: str,
                        lat_column: str,
                        lon_column: str,
                        sheet_name: str = 'Sheet1',
                        skip_rows: int = 0) -> Dict[str, np.ndarray]:

        # Load data from Excel file

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read Excel file
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {str(e)}")

        # Check required columns
        required_columns = [age_column, age_max_column, probability_column, lat_column, lon_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Extract data
        age = df[age_column].values.astype(float)
        age_max = df[age_max_column].values.astype(float)
        probability = df[probability_column].values.astype(float)
        lat = df[lat_column].values.astype(float)
        lon = df[lon_column].values.astype(float)

        # Validate data
        self._validate_data(age, age_max, probability, lat, lon)

        # Calculate age error
        age_error = (age_max - age) / 2

        # Handle missing values
        age, age_error, probability, lat, lon = self._handle_missing_values(
            age, age_error, probability, lat, lon
        )

        return {
            'age': age,
            'age_error': age_error,
            'probability': probability,
            'lat': lat,
            'lon': lon
        }

    def _validate_data(self,
                       age: np.ndarray,
                       age_max: np.ndarray,
                       probability: np.ndarray,
                       lat: np.ndarray,
                       lon: np.ndarray) -> None:

        # Check data types
        if not all(isinstance(arr, np.ndarray) for arr in [age, age_max, probability, lat, lon]):
            raise ValueError("All input data must be numpy arrays")

        # Check array length consistency
        lengths = [len(arr) for arr in [age, age_max, probability, lat, lon]]
        if len(set(lengths)) != 1:
            raise ValueError("All arrays must have the same length")

        # Check age range
        if np.any(age < 0) or np.any(age_max < 0):
            raise ValueError("Age values must not be negative")

        if np.any(age > age_max):
            raise ValueError("Age values must not be greater than max age values")

        # Check probability range
        if np.any(probability < 0) or np.any(probability > 1):
            raise ValueError("Probability values must be between 0 and 1")

        # Check latitude range
        if np.any(lat < -90) or np.any(lat > 90):
            raise ValueError("Latitude values must be between -90 and 90")

        if np.any(lon < -180) or np.any(lon > 180):
            raise ValueError("Longitude values must be between -180 and 180")

    def _handle_missing_values(self,
                               age: np.ndarray,
                               age_error: np.ndarray,
                               probability: np.ndarray,
                               lat: np.ndarray,
                               lon: np.ndarray) -> tuple:

        # Create valid data mask
        valid_mask = (
                ~np.isnan(age) &
                ~np.isnan(age_error) &
                ~np.isnan(probability) &
                ~np.isnan(lat) &
                ~np.isnan(lon)
        )

        # Count missing values
        missing_count = np.sum(~valid_mask)
        if missing_count > 0:
            print(f"Warning: Found {missing_count} missing values, automatically removed")

        # Filter valid data
        age = age[valid_mask]
        age_error = age_error[valid_mask]
        probability = probability[valid_mask]
        lat = lat[valid_mask]
        lon = lon[valid_mask]

        return age, age_error, probability, lat, lon

    def get_data_summary(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:

        # Get data summary information
        return {
            'sample_count': len(data['age']),
            'age_range': (np.min(data['age']), np.max(data['age'])),
            'age_mean': np.mean(data['age']),
            'age_std': np.std(data['age']),
            'probability_mean': np.mean(data['probability']),
            'probability_std': np.std(data['probability']),
            'lat_range': (np.min(data['lat']), np.max(data['lat'])),
            'lon_range': (np.min(data['lon']), np.max(data['lon']))
        }

    def save_processed_data(self,
                            data: Dict[str, np.ndarray],
                            output_file: str) -> bool:

        # Save processed data to Excel file
        try:
            df = pd.DataFrame({
                'age': data['age'],
                'age_error': data['age_error'],
                'probability': data['probability'],
                'lat': data['lat'],
                'lon': data['lon']
            })
            df.to_excel(output_file, index=False)
            print(f"Data saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Failed to save data: {str(e)}")
            return False