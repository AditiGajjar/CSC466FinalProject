import sys
import pandas as pd
from sklearn.model_selection import train_test_split

attributes = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

def clean_dataset(file_path):
    # Load the DataFrame from the CSV file
    df = pd.read_csv(file_path)

    # Select only the specified columns
    selected_columns = attributes + ['playlist_genre']
    df_cleaned = df[selected_columns]

    # Move 'playlist_genre' column to the last position
    df_cleaned = df_cleaned[[col for col in df_cleaned.columns if col != 'playlist_genre'] + ['playlist_genre']]

    return df_cleaned

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file_path> <output_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Clean the dataset
    cleaned_df = clean_dataset(input_file_path)

    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv(output_file_path, index=False)

    train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=42)

    # Save the cleaned training and testing DataFrames to new CSV files
    train_df.to_csv(output_file_path.replace('.csv', '_train.csv'), index=False)
    test_df.to_csv(output_file_path.replace('.csv', '_test.csv'), index=False)

    print(f"Cleaned training data saved to {output_file_path.replace('.csv', '_train.csv')}")
    print(f"Cleaned testing data saved to {output_file_path.replace('.csv', '_test.csv')}")

    #print(f"Cleaned data saved to {output_file_path}")
