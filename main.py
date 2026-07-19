import pandas as pd
from sklearn.model_selection import train_test_split 

from transformers import Outlier_Remover
from config.features import iqr_features, zscore_features, transformations_dict
from config.settings import iqr_threshold, zscore_threshold, outlier_row_threshold_ratio



def main():
    print("Starting Spotify ML Pipeline...")

    data_path = "Database/spotify_songs_initial_cleaned.xlsx"
    
    # Load data:
    try:
        df = pd.read_excel(data_path)
        print(f"[*] Successfully loaded {df.shape[0]} songs.")
    except FileNotFoundError:
        print(f"[!] Error: Could not find {data_path}. Make sure the path is correct.")
        return

    # Separate features and target:
    target_column = 'track_popularity'

    X = df.drop(columns = [target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    print( "-" * 30)
    print(f"[*] Train set size: {X_train.shape[0]} rows (80%)")
    print(f"[*] Test set size: {X_test.shape[0]} rows (20%)")
    print( "-" * 30)

    # Create Outlier Remover:
    outlier_remover = Outlier_Remover(
        iqr_features = iqr_features,
        zscore_features = zscore_features,
        transformations_dict = transformations_dict,
        iqr_threshold = iqr_threshold,
        zscore_threshold = zscore_threshold,
        outlier_row_threshold_ratio = outlier_row_threshold_ratio
    )

    # Remove outliers from TRAINING DATA ONLY:
    X_train_clean, y_train_clean = outlier_remover.fit_transform(X_train, y_train)
    print("-" * 30)
    print(
        f"[*] Cleaned train set size: "
        f"{X_train_clean.shape[0]} rows"
    )
    print(
        f"[*] Removed rows: "
        f"{X_train.shape[0] - X_train_clean.shape[0]}"
    )
    print("-" * 30)

# Executing the main function (responsible for the pipeline of the project):
if __name__ == "__main__":
    main()


    