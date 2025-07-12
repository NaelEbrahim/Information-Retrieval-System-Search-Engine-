import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

class QueryExpander:
    STOPWORDS = set(stopwords.words('english'))

    @classmethod
    def is_valid_term(cls, term):
        """
        Returns True only if term is alphabetic, length >= 3, and not a stopword.
        """
        return (
                len(term) >= 3 and
                term.isalpha() and
                term.lower() not in cls.STOPWORDS
        )

    @classmethod
    def expand_query_terms(cls, query_terms, w2v_model, max_expansions=1, similarity_threshold=0.7):
        """
        Expands query terms using Word2Vec based semantic similarity.
        """
        valid_query_terms = [term for term in query_terms if term in w2v_model.wv]

        if not valid_query_terms:
            return query_terms

        try:
            query_vector = np.mean([w2v_model.wv[term] for term in valid_query_terms], axis=0)
        except Exception as e:
            print(f"Error computing query vector: {e}")
            return query_terms

        expanded_terms = list(set(query_terms))  # Avoid duplicates

        for term in valid_query_terms:
            similar_words = w2v_model.wv.most_similar(term, topn=10)

            added = 0
            for word, score in similar_words:
                if added >= max_expansions:
                    break
                if word in expanded_terms:
                    continue
                if not cls.is_valid_term(word):
                    continue
                try:
                    sim = cosine_similarity([query_vector], [w2v_model.wv[word]])[0][0]
                    if sim >= similarity_threshold:
                        expanded_terms.append(word)
                        added += 1
                except:
                    continue

        return expanded_terms
