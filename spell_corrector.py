import re
import pickle
from collections import Counter
from services.document_service_singleton import DocumentService

class SpellCorrector:
    """
    A simple spell corrector based on Peter Norvig's algorithm.
    """
    def __init__(self, path='spell_corrector.pkl'):
        """Initializes the SpellCorrector, optionally loading a pre-trained model."""
        self.words = Counter()
        self.path = path
        self.is_loaded = self.load()

    def train(self, text):
        """Train the model on a text."""
        self.words.update(self.words_from_text(text))

    def words_from_text(self, text):
        """Extract words from a text."""
        return re.findall(r'\w+', text.lower())

    def probability(self, word):
        """Probability of `word`."""
        total_words = sum(self.words.values())
        if total_words == 0:
            return 0
        return self.words[word] / total_words

    def correction(self, word):
        """Most probable spelling correction for word."""
        return max(self.candidates(word), key=self.probability)

    def candidates(self, word):
        """Generate possible spelling corrections for word."""
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        """The subset of `words` that appear in the dictionary of words."""
        return set(w for w in words if w in self.words)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def check_query(self, query):
        """
        Checks a query and returns a corrected version if any word is misspelled.
        Returns None if the query is already correct.
        """
        if not self.is_loaded:
            print("Spell corrector model is not loaded. Cannot check query.")
            return None
            
        words = self.words_from_text(query)
        corrected_words = [self.correction(word) for word in words]
        corrected_query = " ".join(corrected_words)
        
        # Return the corrected query only if it's different from the original
        return corrected_query if corrected_query != query.lower() else None

    def save(self):
        """Saves the spell corrector model to disk."""
        print(f"Saving spell corrector model to {self.path}...")
        with open(self.path, 'wb') as f:
            pickle.dump(self.words, f)
        print("Model saved successfully.")

    def load(self):
        """Loads the spell corrector model from disk."""
        try:
            print(f"Loading spell corrector model from {self.path}...")
            with open(self.path, 'rb') as f:
                self.words = pickle.load(f)
            print("Model loaded successfully.")
            return True
        except FileNotFoundError:
            print(f"Spell corrector model not found at {self.path}. A new model will be created upon training.")
            return False

def train_spell_corrector():
    """
    Trains the spell corrector on the entire document corpus from the database.
    """
    print("Starting spell corrector training...")
    doc_service = DocumentService()
    documents = doc_service.get_docs_store()

    if not documents:
        print("No documents found in the database. Aborting training.")
        return

    print(f"Training spell corrector on {len(documents)} documents...")
    # Combine all document texts into a single large string
    corpus = " ".join([doc['text'] for doc in documents])
    
    spell_checker = SpellCorrector()
    spell_checker.train(corpus)
    spell_checker.save()
    print("Spell corrector training complete.")

if __name__ == '__main__':
    train_spell_corrector()
