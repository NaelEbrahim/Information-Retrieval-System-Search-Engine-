from flask import Flask, render_template, request
from search_engine import SearchEngine

app = Flask('__main__')

# Initialize the search engine once
search_engine = SearchEngine()

@app.route('/')
def index():
    """Renders the main search page."""
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    """Handles the search query and displays results."""
    query = request.args.get('query')
    dataset = request.args.get('dataset', default='trec-tot/2023/train')
    if not query:
        return render_template('index.html', error="Please enter a search query.")
    
    results = search_engine.search(query, dataset_name=dataset)
    return render_template('results.html', query=query, results=results, dataset=dataset)

@app.route('/document/<doc_id>')
def document(doc_id):
    """Handles the document page."""
    # get datasets from query
    dataset = request.args.get('dataset')
    doc = search_engine.doc_service.get_document(doc_id, dataset)
    if not doc:
        return render_template('not_found.html', doc=doc, dataset=dataset)
    return render_template('document.html', doc=doc, dataset=dataset)

if __name__ == '__main__':
    app.run(debug=True)
