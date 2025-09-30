from flask import Flask, request, render_template, redirect, url_for
import os
from crawler import WebCrawler
from indexer import build_index, save_index, load_index
from search import SearchEngine

app = Flask(__name__)

CORPUS_DIR = "corpus"
INDEX_FILE = "index.json"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/crawl", methods=["GET", "POST"])
def crawl():
    message = ""
    if request.method == "POST":
        url = request.form.get("url")
        max_pages = int(request.form.get("max_pages", 10))
        delay = float(request.form.get("delay", 1.0))
        crawler = WebCrawler(output_dir=CORPUS_DIR, delay=delay)
        stats = crawler.crawl(url, max_pages=max_pages)
        message = f"Crawled {stats['pages_crawled']} pages, {stats['pages_failed']} failed."
    return render_template("crawl.html", message=message)

@app.route("/index", methods=["GET", "POST"])
def index():
    message, stats = "", None
    if request.method == "POST":
        idx, docs = build_index(corpus_dir=CORPUS_DIR)
        if docs:
            save_index(idx, docs, out=INDEX_FILE)
            message = f"Indexed {len(docs)} documents."
            stats = {
                "num_documents": len(docs),
                "num_terms": len(idx),
                "total_tokens": sum(len(postings) for term_docs in idx.values() for postings in term_docs.values())
            }
        else:
            message = "No .txt files found to index."
    return render_template("index.html", message=message, stats=stats)

@app.route("/search", methods=["GET", "POST"])
def search():
    results, stats, query = [], {}, ""
    if os.path.exists(INDEX_FILE):
        engine = SearchEngine(index_file=INDEX_FILE)
    else:
        engine = None

    if request.method == "POST" and engine is not None:
        query = request.form.get("query", "")
        if query:
            if query.startswith('"') and query.endswith('"'):
                phrase_terms = engine.tokenize(query[1:-1])
                matches = engine.find_phrase_matches(phrase_terms)
                for doc in matches:
                    snippet = engine.generate_snippet(doc, phrase_terms)
                    results.append({"doc": doc, "snippet": snippet, "score": None})
            else:
                terms = engine.tokenize(query)
                scores = engine.calculate_tf_idf_scores(terms)
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for doc, score in ranked:
                    snippet = engine.generate_snippet(doc, terms)
                    results.append({"doc": doc, "snippet": snippet, "score": round(score, 3)})
        stats = engine.get_statistics()
    return render_template("search.html", results=results, stats=stats, query=query)

if __name__ == "__main__":
    app.run(debug=True)
