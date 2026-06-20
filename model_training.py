import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import local preprocessing modules
from data_preprocessing import load_data, preprocess_text
from feature_extraction import get_tfidf_vectorizer

def train_models():
    # 1. Load Data
    print("Loading data...")
    df = load_data("dataset/spam.csv")
    
    # 2. Preprocess Data
    print("Preprocessing messages (this may take a few seconds)...")
    df['cleaned_message'] = df['message'].apply(preprocess_text)
    
    # Drop rows with empty messages after cleaning (if any)
    df = df[df['cleaned_message'] != '']
    
    X = df['cleaned_message']
    y = df['label_num']
    
    # 3. Train-Test Split (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # 4. Define Models to Compare
    models = {
        'Naive Bayes (MultinomialNB)': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    
    # 5. Train and Evaluate each Model
    for model_name, classifier in models.items():
        print(f"Training {model_name}...")
        
        # Build pipeline
        pipeline = Pipeline([
            ('tfidf', get_tfidf_vectorizer(max_features=3000)),
            ('classifier', classifier)
        ])
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        
        results.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Make sure output folder exists
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    results_df.to_csv("outputs/model_comparison.csv", index=False)
    print("Saved model comparison table to 'outputs/model_comparison.csv'")
    
    # 6. Select and Save the Best Model
    # We will choose based on F1-Score (since spam datasets are imbalanced)
    best_row = results_df.loc[results_df['F1-Score'].idxmax()]
    best_model_name = best_row['Model']
    print(f"\nBest Model selected: {best_model_name} (F1-Score: {best_row['F1-Score']:.4f})")
    
    best_pipeline = trained_pipelines[best_model_name]
    
    # Save the pipeline to pickle
    model_path = "models/spam_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
        
    print(f"Saved best model pipeline to '{model_path}'")
    
    # Save split test data for evaluation script
    test_data = pd.DataFrame({'message': X_test, 'label': y_test})
    test_data.to_csv("outputs/test_data.csv", index=False)
    
    return best_model_name, best_pipeline

if __name__ == "__main__":
    train_models()
