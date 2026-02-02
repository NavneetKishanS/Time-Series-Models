"""
Analyzes the preprocessed data to understand the relationship between
conditioning features and body regions.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def decode_bodygroup(encoded_value):
    """Decodes body group ID to text."""
    bodygroup_map = {
        0: 'HEAD',
        1: 'NECK',
        2: 'CHEST',
        3: 'ABDOMEN',
        4: 'PELVIS',
        5: 'SPINE',
        6: 'LOWER_EXTREMITY',
        7: 'UPPER_EXTREMITY',
        8: 'SHOULDER',
        9: 'UNKNOWN_REGION',
        10: 'UNKNOWN'
    }
    return bodygroup_map.get(encoded_value, 'UNKNOWN')

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/preprocessed/all_preprocessed.csv')
    print("Data loaded successfully.")

    df['BodyGroup_from_text'] = df['BodyGroup_from'].apply(decode_bodygroup)

    print("\n" + "="*80)
    print("Correlation matrix of numerical conditioning features")
    print("="*80)
    numerical_features = ['Age', 'Weight', 'Height', 'BodyGroup_from', 'BodyGroup_to', 'PTAB', 'entity_type']
    corr_matrix = df[numerical_features].corr()
    print(corr_matrix.round(2))

    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Conditioning Features')
    plt.savefig('visualizations/correlation_matrix.png')
    print("\nCorrelation matrix plot saved to visualizations/correlation_matrix.png")

if __name__ == "__main__":
    main()
