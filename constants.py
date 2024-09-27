

import os
import nltk
from nltk.corpus import stopwords

# Download Turkish stopwords (only needed once)
nltk.download('stopwords')

# Initialize Turkish stopwords
turkish_stopwords = set(stopwords.words('turkish'))


# Paths
# Update this path to your actual "sample" folder location
samples_folder = r"C:\Users\gampa\OneDrive\Masaüstü\sample"

# Database directory
db_directory = os.path.join(os.getcwd(), "Advanced-File-Search-main", "databases")

# Create database directory if it doesn't exist
if not os.path.exists(db_directory):
    os.makedirs(db_directory)

# Model parameters
dimension = 768  # Embedding dimension for the model
