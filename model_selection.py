# model_selection.py

def is_short_query(query):
    return len(query.strip().split()) <= 2

def is_complex_query(query):
    question_words = ['ne', 'neden', 'nasıl', 'ne zaman', 'nerede', 'kim', 'hangi', 'kimin', 'kaç']
    query_lower = query.lower()
    return (any(query_lower.startswith(word) for word in question_words) or
            query_lower.endswith('?') or
            len(query.strip().split()) > 5)

def select_model(query):
    if is_complex_query(query):
        return 'sbert'
    elif is_short_query(query):
        return 'tfidf'
    else:
        return 'combined'

def get_similarity_weights(query):
    if is_complex_query(query):
        return 0.2, 0.8  # Favor SBERT
    elif is_short_query(query):
        return 0.8, 0.2  # Favor TF-IDF
    else:
        return 0.5, 0.5  # Equal weights
