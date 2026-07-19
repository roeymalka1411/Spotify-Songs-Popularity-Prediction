import pandas as pd
import numpy as np
from scipy import stats

class Outlier_Remover:
    """
        A custom transformer that applies temporary mathematical transformations to features,
        identifies outliers using IQR and Z-score on the transformed data, and removes 
        the corresponding problematic rows from the original dataset.
    """

    def __init__(self, iqr_features, zscore_features, transformations_dict = None, iqr_threshold = 1.5, zscore_threshold = 3.0, outlier_row_threshold_ratio = 0.2):
        self.iqr_features = iqr_features
        self.zscore_features = zscore_features
        self.iqr_threshold = iqr_threshold
        self.zscore_threshold = zscore_threshold
        self.outlier_row_threshold_ratio = outlier_row_threshold_ratio
        self.transformations_dict = transformations_dict if transformations_dict else {}

    def apply_transformations(self, data):
        """
            Applies temporary transformations (e.g., Log, Yeo-Johnson) to a copy of the data.
            This approximates a normal distribution to make methods like Z-score mathematically valid.
        """
        
        x = data.copy()

        def apply_shift(feature_data):
            min_val = feature_data.min()
            shift = (abs(min_val) + 0.001) if min_val <= 0 else 0
            return feature_data + shift

        for key, features_list in self.transformations_dict.items():
            for feature in features_list:
                if feature in x.columns:                    
                    if key == 'yj_transform':
                        x[feature] = stats.yeojohnson(x[feature])[0] # [0] responsible for the data.
                    elif key == 'expo_transform':
                        x[feature] = x[feature] ** 2
                    elif key == 'log_transform':
                        shifted_data = apply_shift(x[feature])
                        x[feature] = np.log(shifted_data)
                    elif key == 'sqrt_transform':
                        shifted_data = apply_shift(x[feature])
                        x[feature] = np.sqrt(shifted_data)
        
        return x
    
    def get_IQR_mask(self, data):
        """
            Evaluates the temporarily transformed data using the Interquartile Range (IQR) method.
            Returns a boolean mask where True indicates an outlier.
        """
        x = data[self.iqr_features].copy()
        mask = pd.DataFrame(index = x.index)

        for feature in self.iqr_features:
            Q1 = x[feature].quantile(0.25)
            Q3 = x[feature].quantile(0.75)
            IQR_val = Q3 - Q1

            lower_bound = Q1 - (self.iqr_threshold * IQR_val)
            upper_bound = Q3 + (self.iqr_threshold * IQR_val)

            mask[feature] = (x[feature] < lower_bound) | (x[feature] > upper_bound)
        
        return mask

    def get_Zscore_mask(self, data):
        """
            Evaluates the temporarily transformed data using the standard Z-score method.
            Returns a boolean mask where True indicates an outlier.
        """
        x = data[self.zscore_features].copy()
        mask = pd.DataFrame(index = x.index)

        for feature in self.zscore_features:
            mean = x[feature].mean()
            std = x[feature].std()
            zscores = (x[feature] - mean) / std

            mask[feature] = abs(zscores) > self.zscore_threshold
        
        return mask

    def fit_transform(self, X, y):
        """
            Executes the pipeline: applies temp transformations, generates IQR/Z-score masks, 
            and drops rows exceeding the outlier threshold from the original X and y data.
        """

        x_temp = X.copy()

        x_temp = self.apply_transformations(x_temp)
        iqr_mask = self.get_IQR_mask(x_temp)
        zscore_mask = self.get_Zscore_mask(x_temp)

        all_outliers_mask = pd.concat([iqr_mask, zscore_mask], axis = 1)

        total_outliers_per_row = all_outliers_mask.sum(axis = 1)
        
        total_features = len(self.iqr_features) + len(self.zscore_features)
        min_outliers_to_drop = total_features * self.outlier_row_threshold_ratio

        bad_indices = all_outliers_mask[total_outliers_per_row >= min_outliers_to_drop].index

        X_clean = X.drop(index = bad_indices)
        y_clean = y.drop(index = bad_indices)

        print(f"OutlierRemover dropped {len(bad_indices)} rows.")

        return X_clean, y_clean










        

                
        






    
    