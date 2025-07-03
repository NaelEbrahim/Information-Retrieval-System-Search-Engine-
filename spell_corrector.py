from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

class SpellCorrector:
    def __init__(self):
        self.spell_checker = SpellChecker()

    def correct_sentence_spelling(self, query):
        query_tokens = word_tokenize(query.lower())
        misspelled = self.spell_checker.unknown(query_tokens)
        corrected_tokens = []
        for token in query_tokens:
            if token in misspelled:
                corrected_token = self.spell_checker.correction(token)
                if corrected_token is None:
                    corrected_token = token  
            else:
                corrected_token = token
            corrected_tokens.append(corrected_token)
        
        return ' '.join(corrected_tokens)


if __name__ == '__main__':
    spell_corrector = SpellCorrector()
    query = "whaat are in the Suddim hussai in eaath"
    corrected_query = spell_corrector.correct_sentence_spelling(query)
    print(f"Corrected query: '{corrected_query}'")