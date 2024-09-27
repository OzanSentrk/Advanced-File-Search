

import re
from constants import turkish_stopwords

# Preprocessing function (without stemming)
def preprocess_text(text):
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = text.split()
    # Remove Turkish stopwords
    tokens = [word for word in tokens if word not in turkish_stopwords]
    # Do not apply stemming
    cleaned_text = ' '.join(tokens)
    return cleaned_text
