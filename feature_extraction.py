from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(max_features=3000):
    """
    Initialize and return a TfidfVectorizer.
    We configure it with max_features to limit the vocabulary size to the most important words.
    """
    return TfidfVectorizer(max_features=max_features)
