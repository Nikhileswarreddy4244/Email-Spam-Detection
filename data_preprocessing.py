import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def download_nltk_resources():
    """Ensure NLTK resources are downloaded."""
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

# Run download on import to ensure availability
download_nltk_resources()

def load_data(filepath="dataset/spam.csv"):
    """Load the raw dataset and clean the column structure."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Load dataset with latin-1 encoding as it contains special characters
    df = pd.read_csv(filepath, encoding='latin-1')
    
    # Drop unnamed columns if they exist
    cols_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    # Rename columns to standard names
    if len(df.columns) >= 2:
        df.columns = ['label', 'message'] + list(df.columns[2:])
        
    # Keep only label and message
    df = df[['label', 'message']]
    
    # Map label to binary target: ham -> 0, spam -> 1
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df

def preprocess_text(text):
    """
    Clean and preprocess a single message:
    1. Lowercase
    2. Keep only letters (remove punctuation/numbers)
    3. Tokenize
    4. Remove stopwords
    5. Apply Porter Stemming
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Keep only alphabetical characters (replace punctuation/numbers with spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenize using split (safer and faster than nltk tokenizers for this dataset)
    words = text.split()
    
    # Get English stopwords list
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        download_nltk_resources()
        stop_words = set(stopwords.words('english'))
        
    ps = PorterStemmer()
    
    # Filter stopwords and apply stemming
    cleaned_words = [ps.stem(word) for word in words if word not in stop_words]
    
    # Join back into a single string
    return " ".join(cleaned_words)

if __name__ == "__main__":
    # Quick self-test
    download_nltk_resources()
    print("Loading data sample...")
    try:
        data = load_data()
        print("Data loaded successfully. Shape:", data.shape)
        print("Preprocessing test...")
        sample_msg = "Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
        print("Original:", sample_msg)
        print("Preprocessed:", preprocess_text(sample_msg))
    except Exception as e:
        print("Error during test:", e)
