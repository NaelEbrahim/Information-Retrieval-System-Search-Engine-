from flask import Flask, render_template, request
from ir_project.search_engine import SearchEngine

app = Flask(__name__)

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
    if not query:
        return render_template('index.html', error="Please enter a search query.")
    
    results = search_engine.search(query)
    
    return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
