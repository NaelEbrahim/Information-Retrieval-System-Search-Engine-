from flask import Flask, render_template, request
# from search_engine import SearchEngine

app = Flask(__name__)

# Initialize the search engine once
# search_engine = SearchEngine()

@app.route('/')
def index():
    """Renders the main search page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the search query and displays results."""
    query = request.form.get('query')
    if not query:
        return render_template('index.html', error="Please enter a search query.")
    
    # results = search_engine.search(query)
    results = [
        {
            'result_id': 1,
            'title': 'Document 1',
            'snippet': 'This is the first document.',
            'result_detail': 'sfsdf'
        },
        {
            'result_id': 2,
            'title': 'Document 2',
            'snippet': 'This is the second document.',
            'result_detail': 'sfsdf'
        }
    ]
    return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
