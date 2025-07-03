from flask import Flask, render_template, request, jsonify
from search_engine import SearchEngine
from services.query_suggestion_service import QuerySuggestionService

app = Flask('__main__')

# Initialize the search engine once
search_engine = SearchEngine()
query_suggestion_service = QuerySuggestionService()

@app.route('/')
def index():
    """Renders the main search page."""
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    """Handles the search query and displays results."""
    query = request.args.get('query')
    dataset = request.args.get('dataset', default='trec-tot/2023/train')
    model_type = request.args.get('model_type', default='tfidf')
    if not query:
        return render_template('index.html', error="Please enter a search query.")
    
    num_results, results = search_engine.search(query, dataset_name=dataset, model_type=model_type)
    return render_template('results.html', query=query, results=results, dataset=dataset, model_type=model_type, num_results=num_results)

@app.route('/document/<doc_id>')
def document(doc_id):
    """Handles the document page."""
    # get datasets from query
    dataset = request.args.get('dataset')
    doc = search_engine.doc_service.get_document(doc_id, dataset)
    if not doc:
        return render_template('not_found.html', doc=doc, dataset=dataset)
    return render_template('document.html', doc=doc, dataset=dataset)

@app.route('/suggest')
def suggest():
    """Returns a list of query suggestions."""
    query = request.args.get('query', '')
    dataset = request.args.get('dataset', default='trec-tot/2023/train')
    suggestions = query_suggestion_service.get_suggestions(query, dataset_name=dataset)
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)
