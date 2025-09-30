import os, re, json
from collections import defaultdict
from tqdm import tqdm  # Progress bar for large corpora

CORPUS_DIR = "corpus"
INDEX_FILE = "index.json"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Extract lowercase alphanumeric tokens from text"""
    return TOKEN_RE.findall(text.lower())


def build_index(corpus_dir=CORPUS_DIR):
    """Build inverted index from text files in corpus directory"""
    index = defaultdict(lambda: defaultdict(list))  # term -> doc -> [positions]
    documents = []

    # Get all .txt files
    txt_files = [f for f in sorted(os.listdir(corpus_dir)) if f.endswith(".txt")]

    if not txt_files:
        print(f"Warning: No .txt files found in {corpus_dir}")
        return {}, []

    # Process each file with progress bar
    for fname in tqdm(txt_files, desc="Indexing documents"):
        path = os.path.join(corpus_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        tokens = tokenize(text)
        documents.append(fname)

        # Build positional index
        for pos, tok in enumerate(tokens):
            index[tok][fname].append(pos)

    # Convert to regular dicts for JSON serialization
    index_plain = {term: dict(postings) for term, postings in index.items()}
    return index_plain, documents


def save_index(index, documents, out=INDEX_FILE):
    """Save index and document list to JSON file"""
    data = {
        "index": index,
        "documents": documents,
        "stats": {
            "num_documents": len(documents),
            "num_terms": len(index),
            "total_tokens": sum(len(postings) for term_docs in index.values()
                                for postings in term_docs.values())
        }
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved index to {out}")
    print(f"📄 Documents: {len(documents)}")
    print(f"🔤 Unique terms: {len(index)}")
    print(f"📊 Total tokens indexed: {data['stats']['total_tokens']}")


def load_index(index_file=INDEX_FILE):
    """Load index from JSON file"""
    try:
        with open(index_file, encoding="utf-8") as f:
            data = json.load(f)
        return data["index"], data["documents"]
    except FileNotFoundError:
        print(f"Index file {index_file} not found. Run indexer first.")
        return {}, []


if __name__ == "__main__":
    # Create corpus directory if it doesn't exist
    os.makedirs(CORPUS_DIR, exist_ok=True)

    print(f"🔍 Building search index from {CORPUS_DIR}/")
    idx, docs = build_index()

    if docs:
        save_index(idx, docs)
    else:
        print("❌ No documents found to index!")
        print(f"💡 Add some .txt files to the {CORPUS_DIR}/ directory first.")