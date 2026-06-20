import os
import argparse
import pickle

# Import preprocessing
from data_preprocessing import preprocess_text

def load_classifier(model_path="models/spam_classifier.pkl"):
    """Load the trained model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run model_training.py first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_message(pipeline, message):
    """Clean the message and predict its category (Ham vs Spam)."""
    cleaned = preprocess_text(message)
    if not cleaned:
        # If message becomes empty after cleaning (e.g. only punctuation)
        # return ham as default with high uncertainty
        return "ham", 0.50
        
    # Predict probabilities (if supported)
    proba = pipeline.predict_proba([cleaned])[0]
    prediction = pipeline.predict([cleaned])[0]
    
    label = "spam" if prediction == 1 else "ham"
    confidence = proba[prediction]
    
    return label, confidence

def main():
    parser = argparse.ArgumentParser(description="Classify an SMS/Email message as Ham or Spam.")
    parser.add_argument("--message", type=str, help="The message content to classify.")
    args = parser.parse_args()
    
    try:
        pipeline = load_classifier()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by running 'python src/model_training.py'")
        return
        
    if args.message:
        # Single message mode
        label, confidence = predict_message(pipeline, args.message)
        print("\n" + "="*50)
        print(f"Message: {args.message}")
        print(f"Result : {label.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("="*50 + "\n")
    else:
        # Interactive mode
        print("\n--- Email/SMS Spam Detector CLI (Interactive Mode) ---")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                message = input("Enter a message to classify: ").strip()
                if not message:
                    continue
                if message.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                    
                label, confidence = predict_message(pipeline, message)
                
                print("-"*50)
                print(f"Prediction: {label.upper()}")
                print(f"Confidence: {confidence * 100:.2f}%")
                print("-"*50 + "\n")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
