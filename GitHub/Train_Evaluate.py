import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    # Paths to the merged CSV files
    merged_training_byarticle_csv = r'C:\Users\PTGHYD\Downloads\Train_Model\merged_train_byarticle.csv'
    merged_training_bypublisher_1_csv = r'C:\Users\PTGHYD\Downloads\Train_Model\merged_train_bypublisher_1.csv'
    merged_validation_bypublisher_1_csv = r'C:\Users\PTGHYD\Downloads\Train_Model\merged_validation_bypublisher_1.csv'

    # Verify if files exist and are not empty
    for file_path in [merged_training_byarticle_csv, merged_training_bypublisher_1_csv, merged_validation_bypublisher_1_csv]:
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if not first_line:
                    raise ValueError(f"The file {file_path} is empty.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        except ValueError as ve:
            raise ve

    # Load the merged DataFrames
    df_training_byarticle = pd.read_csv(merged_training_byarticle_csv)
    df_training_bypublisher_1 = pd.read_csv(merged_training_bypublisher_1_csv)
    df_validation_bypublisher_1 = pd.read_csv(merged_validation_bypublisher_1_csv)

    return df_training_byarticle, df_training_bypublisher_1, df_validation_bypublisher_1

def prepare_features_and_labels(df_training_byarticle, df_training_bypublisher_1, df_validation_bypublisher_1):
    # Debug: print column names to check for discrepancies
    print("Columns in df_training_byarticle:", df_training_byarticle.columns)
    print("Columns in df_training_bypublisher_1:", df_training_bypublisher_1.columns)
    print("Columns in df_validation_bypublisher_1:", df_validation_bypublisher_1.columns)

    # Check if 'label' is present in all DataFrames
    for df, name in [(df_training_byarticle, 'df_training_byarticle'), 
                     (df_training_bypublisher_1, 'df_training_bypublisher_1'), 
                     (df_validation_bypublisher_1, 'df_validation_bypublisher_1')]:
        if 'label' not in df.columns:
            raise KeyError(f"Column 'label' not found in {name}")

    # Extract features and labels from the training DataFrame
    X_train_byarticle = df_training_byarticle.drop(columns=['article_id', 'label'])
    y_train_byarticle = df_training_byarticle['label']

    X_train_bypublisher_1 = df_training_bypublisher_1.drop(columns=['article_id', 'label'])
    y_train_bypublisher_1 = df_training_bypublisher_1['label']

    X_val_bypublisher_1 = df_validation_bypublisher_1.drop(columns=['article_id', 'label'])
    y_val_bypublisher_1 = df_validation_bypublisher_1['label']

    return X_train_byarticle, y_train_byarticle, X_train_bypublisher_1, y_train_bypublisher_1, X_val_bypublisher_1, y_val_bypublisher_1

def align_columns(*dfs):
    # Get the union of all columns
    all_columns = sorted(set(col for df in dfs for col in df.columns))

    # Process dataframes in chunks to avoid memory errors
    aligned_dfs = []
    for df in dfs:
        aligned_dfs.append(pd.concat([chunk.reindex(columns=all_columns, fill_value=0) for chunk in np.array_split(df, 10)], ignore_index=True))

    return aligned_dfs

def train_and_evaluate(X_train, y_train, X_val1, y_val1):
    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the first validation set
    y_pred1 = model.predict(X_val1)

    # Evaluate the model on the first validation set
    accuracy1 = accuracy_score(y_val1, y_pred1)
    precision1 = precision_score(y_val1, y_pred1, average='weighted')  # Use 'weighted' for multiclass
    recall1 = recall_score(y_val1, y_pred1, average='weighted')        # Use 'weighted' for multiclass
    f1_1 = f1_score(y_val1, y_pred1, average='weighted')               # Use 'weighted' for multiclass

    print(f"Validation Set 1 - Accuracy: {accuracy1}")
    print(f"Validation Set 1 - Precision: {precision1}")
    print(f"Validation Set 1 - Recall: {recall1}")
    print(f"Validation Set 1 - F1 Score: {f1_1}")

def main():
    # Load the data
    df_training_byarticle, df_training_bypublisher_1, df_validation_bypublisher_1 = load_data()

    # Prepare features and labels
    X_train_byarticle, y_train_byarticle, X_train_bypublisher_1, y_train_bypublisher_1, X_val_bypublisher_1, y_val_bypublisher_1 = prepare_features_and_labels(
        df_training_byarticle, df_training_bypublisher_1, df_validation_bypublisher_1)

    # Align columns for all training and validation sets
    X_train_byarticle, X_val_bypublisher_1 = align_columns(
        X_train_byarticle, X_val_bypublisher_1)

    # Train and evaluate using training by article data
    print("Training and evaluating with training by article data:")
    train_and_evaluate(X_train_byarticle, y_train_byarticle, X_val_bypublisher_1, y_val_bypublisher_1)

    # Align columns for training by publisher data
    X_train_bypublisher_1, X_val_bypublisher_1 = align_columns(
        X_train_bypublisher_1, X_val_bypublisher_1)

    # Train and evaluate using training by publisher data
    print("\nTraining and evaluating with training by publisher data:")
    train_and_evaluate(X_train_bypublisher_1, y_train_bypublisher_1, X_val_bypublisher_1, y_val_bypublisher_1)

if __name__ == '__main__':
    main()
