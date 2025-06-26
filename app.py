from flask import Flask, render_template, request
from search_engine import SearchEngine

app = Flask('__main__')

# Initialize the search engine once
search_engine = SearchEngine()

@app.route('/')
def index():
    """Renders the main search page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the search query and displays results."""
    query = request.form.get('query')
    dataset = request.form.get('dataset')
    if not query:
        return render_template('index.html', error="Please enter a search query.")

    if not dataset and dataset != 'trec-tot/2023/train' and dataset != 'antique/train':
        dataset = 'trec-tot/2023/train'
    
    results = search_engine.search(query, dataset_name=dataset)
    return render_template('results.html', query=query, results=results, dataset=dataset)

@app.route('/document/<doc_id>')
def document(doc_id):
    """Handles the document page."""
    doc = search_engine.doc_service.get_document(doc_id)
    return render_template('document.html', doc=doc)

if __name__ == '__main__':
    app.run(debug=True)
