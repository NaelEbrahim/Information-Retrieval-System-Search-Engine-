import sys
import os
from itertools import product
from difflib import get_close_matches

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.word2vec_singleton_service import Word2VecSingletonService
from preprocessor import Preprocessor

class QuerySuggestionService:
    def __init__(self):
        """Initializes the Query Suggestion Service."""
        self.w2v_singleton_service = Word2VecSingletonService()
        self.preprocessor = Preprocessor()

    def get_suggestions(self, query, dataset_name, top_n=3):
        """
        Generates a list of query suggestions based on semantic and syntactic analysis.
        """
        if not query:
            return []

        processed_tokens = self.preprocessor.process(query)
        
        w2v_service = self.w2v_singleton_service.get_word2vec_service(dataset_name)
        w2v_model = w2v_service.get_model()

        if w2v_model is None:
            print(f"Word2Vec model for {dataset_name} is not loaded.")
            return []

        term_suggestions = self._suggest_terms(processed_tokens, w2v_model, top_n)

        if not term_suggestions:
            return []

        # Generate query combinations
        combinations = list(product(*term_suggestions))
        
        # Format and limit the suggestions
        suggested_queries = [' '.join(combo) for combo in combinations]
        
        return suggested_queries[:top_n]

    def _suggest_terms(self, query_tokens, w2v_model, top_n):
        """
        For each token in the query, generate a list of suggested replacement terms.
        """
        all_suggestions = []
        vocab = list(w2v_model.wv.key_to_index.keys())

        for token in query_tokens:
            suggestions = []
            if token in w2v_model.wv:
                # Term is in vocabulary, find semantically similar terms
                suggestions.append(token)
                similar_terms = w2v_model.wv.most_similar(token, topn=top_n - 1)
                suggestions.extend([term for term, _ in similar_terms])
            else:
                # Term is not in vocabulary, find syntactically similar terms (typo correction)
                close_matches = get_close_matches(token, vocab, n=top_n, cutoff=0.8)
                if close_matches:
                    suggestions.extend(close_matches)

            if suggestions:
                all_suggestions.append(suggestions)
        
        return all_suggestions

if __name__ == '__main__':
    # Example Usage (requires trained models)
    suggestion_service = QuerySuggestionService()
    
    # You need to specify a dataset that has a trained Word2Vec model
    # For example: 'antique/train' or 'trec-tot/train'
    dataset = "antique/train" 
    
    test_queries = ["computr", "scince", "machine learnin"]
    for q in test_queries:
        print(f"Query: '{q}'")
        suggestions = suggestion_service.get_suggestions(q, dataset_name=dataset)
        print(f"Suggestions: {suggestions}\n")