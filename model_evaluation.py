import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model():
    print("Evaluating saved model...")
    
    # 1. Load best model pipeline
    model_path = "models/spam_classifier.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run model_training.py first.")
        
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
        
    # 2. Load test data
    test_data_path = "outputs/test_data.csv"
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}. Run model_training.py first.")
        
    test_df = pd.read_csv(test_data_path)
    # Fill empty clean messages if any
    test_df['message'] = test_df['message'].fillna("")
    
    X_test = test_df['message']
    y_test = test_df['label']
    
    # 3. Predict
    y_pred = pipeline.predict(X_test)
    
    # 4. Generate Classification Report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    print(report)
    
    # Save classification report to txt file
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)
        
    # 5. Generate and Save Confusion Matrix Plot
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    
    cm_path = "outputs/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to '{cm_path}'")
    
    # 6. Generate and Save Accuracy/Metric Comparison Plot
    comparison_path = "outputs/model_comparison.csv"
    if os.path.exists(comparison_path):
        print("Generating Model Comparison Graph...")
        comp_df = pd.read_csv(comparison_path)
        
        # Melt dataframe for plotting multiple metrics
        melted_df = pd.melt(comp_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df, palette='Set2')
        plt.title('Model Performance Comparison', fontsize=14, pad=15)
        plt.ylim(0.7, 1.02)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        graph_path = "outputs/accuracy_graph.png"
        plt.savefig(graph_path, dpi=300)
        plt.close()
        print(f"Saved performance comparison graph to '{graph_path}'")
        
if __name__ == "__main__":
    evaluate_model()
