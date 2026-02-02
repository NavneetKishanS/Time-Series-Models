"""
Trains a simple classifier to predict the body region from conditioning features.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/preprocessed/all_preprocessed.csv')
    print("Data loaded successfully.")

    # Prepare data
    df = df.drop_duplicates(subset=['SeqOrder'])
    X = df[['Age', 'Weight', 'Height', 'PTAB', 'entity_type']]
    y = df['BodyGroup_from']

    # Handle missing values
    X = X.fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    print("\nTraining Logistic Regression classifier...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

