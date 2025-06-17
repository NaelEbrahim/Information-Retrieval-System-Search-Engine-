import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

def download_nltk_data():
    """Downloads necessary NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')

class Preprocessor:
    def __init__(self):
        """Initializes the Preprocessor."""
        download_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def process(self, text):
        """
        Performs the full preprocessing pipeline on a given text.
        1. Tokenization & Normalization (lowercase, punctuation removal)
        2. Stop word removal
        3. Stemming
        """
        # Convert to lower case
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        # Stemming
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        return stemmed_tokens

if __name__ == '__main__':
    # This block will be executed when the script is run directly
    # It ensures that the necessary NLTK data is downloaded
    print("Initializing preprocessor and downloading NLTK data if necessary...")
    preprocessor = Preprocessor()
    print("Preprocessor initialized.")

    # Example usage
    sample_text = "This is a sample sentence, showing off the stop words filtration and stemming."
    processed_tokens = preprocessor.process(sample_text)
    print(f"\nOriginal Text: {sample_text}")
    print(f"Processed Tokens: {processed_tokens}")
