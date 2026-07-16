import pandas as pd
from sklearn.model_selection import train_test_split 

def main():
    print("Starting Spotify ML Pipeline...")

    data_path = "Database/spotify_songs_initial_cleaned.xlsx"
    
    # Trying to load the data:
    try:
        df = pd.read_excel(data_path)
        print(f"[*] Successfully loaded {df.shape[0]} songs.")
    except FileNotFoundError:
        print(f"[!] Error: Could not find {data_path}. Make sure the path is correct.")
        return

    target_column = 'track_popularity'

    X = df.drop(columns = [target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    print( "-" * 30)
    print(f"[*] Train set size: {X_train.shape[0]} rows (80%)")
    print(f"[*] Test set size: {X_test.shape[0]} rows (20%)")
    print( "-" * 30)

# Executing the main function (responsible for the pipeline of the project):
if __name__ == "__main__":
    main()


    