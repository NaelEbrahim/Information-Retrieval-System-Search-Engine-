from flask import Flask, render_template, request, jsonify
from services.retrieval.document_service_singleton import DocumentService
from services.search.search_engine import SearchEngine
from services.helpers.query_suggestion_service import QuerySuggestionService

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
    dataset = request.args.get('dataset', default='trec')
    model_type = request.args.get('model_type', default='tfidf')
    if (dataset not in DocumentService.available_datasets):
        return render_template('index.html', error="Invalid dataset name.")
    if (model_type not in ['tfidf', 'word2vec', 'hybrid', 'search_with_vector_store']):
        return render_template('index.html', error="Invalid model type.")

    try:
        top_n = int(request.args.get('top_n', default=10))
    except ValueError:
        top_n = 10

    if not query:
        return render_template('index.html', error="Please enter a search query.")
    
    num_results, results = search_engine.search(query, dataset_name=dataset, model_type=model_type, top_n=top_n)
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
    dataset = request.args.get('dataset', default='trec')
    if (dataset not in DocumentService.available_datasets):
        return jsonify([])
    suggestions = query_suggestion_service.get_suggestions(query, dataset_name=dataset)
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=False)
