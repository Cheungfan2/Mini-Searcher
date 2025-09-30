#!/usr/bin/env python3
"""
Enhanced document search engine with TF-IDF scoring and phrase matching.

This module provides a command-line search interface for indexed documents,
supporting both keyword searches (ranked by TF-IDF) and exact phrase matching.
"""

import json
import math
import sys
import re
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

# Configuration
INDEX_FILE = "index.json"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SNIPPET_CONTEXT_BEFORE = 60
SNIPPET_CONTEXT_AFTER = 140
SNIPPET_FALLBACK_LENGTH = 200
MAX_RESULTS_DISPLAY = 10


class SearchEngine:
    """Main search engine class handling index loading and query processing."""

    def __init__(self, index_file: str = INDEX_FILE):
        """Initialize the search engine with an index file."""
        self.index_file = index_file
        self.index = {}
        self.documents = []
        self.load_index()

    def load_index(self) -> None:
        """Load the inverted index and document list from JSON file."""
        try:
            with open(self.index_file, encoding="utf-8") as f:
                data = json.load(f)
            self.index = data["index"]
            self.documents = data["documents"]
            print(f"✓ Loaded index with {len(self.documents)} documents and {len(self.index)} terms")
        except FileNotFoundError:
            print(f"Error: Index file '{self.index_file}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in index file: {e}")
            sys.exit(1)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase alphanumeric tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of token strings
        """
        return TOKEN_PATTERN.findall(text.lower())

    def calculate_tf_idf_scores(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for documents given query terms.

        Args:
            query_terms: List of query tokens

        Returns:
            Dictionary mapping document names to their TF-IDF scores
        """
        num_docs = len(self.documents)
        scores = defaultdict(float)

        for term in query_terms:
            if term not in self.index:
                continue

            postings = self.index[term]
            doc_freq = len(postings)

            if doc_freq == 0:
                continue

            # Calculate IDF with smoothing
            idf = math.log((num_docs + 1) / (doc_freq + 1))

            # Calculate TF for each document and accumulate scores
            for doc_name, positions in postings.items():
                term_freq = len(positions)
                # Could normalize by document length here if needed
                scores[doc_name] += term_freq * idf

        return dict(scores)

    def find_phrase_matches(self, phrase_terms: List[str]) -> List[str]:
        """
        Find documents containing an exact phrase (consecutive terms).

        Args:
            phrase_terms: List of tokens forming the phrase

        Returns:
            List of document names containing the exact phrase
        """
        if not phrase_terms:
            return []

        # Find candidate documents that contain all phrase terms
        candidate_docs = None
        for term in phrase_terms:
            if term not in self.index:
                return []  # Term not in index, no matches possible

            docs_with_term = set(self.index[term].keys())
            if candidate_docs is None:
                candidate_docs = docs_with_term
            else:
                candidate_docs &= docs_with_term

        if not candidate_docs:
            return []

        # Check for consecutive positions in candidate documents
        matches = []
        for doc in candidate_docs:
            if self._has_consecutive_terms(doc, phrase_terms):
                matches.append(doc)

        return matches

    def _has_consecutive_terms(self, doc: str, terms: List[str]) -> bool:
        """Check if document has terms in consecutive positions."""
        # Get position lists for all terms in this document
        position_sets = [set(self.index[term][doc]) for term in terms]

        # Check if any starting position has all subsequent positions
        for start_pos in self.index[terms[0]][doc]:
            if all((start_pos + i) in position_sets[i] for i in range(1, len(terms))):
                return True

        return False

    def generate_snippet(self, doc_file: str, highlight_terms: List[str],
                         max_length: int = 200) -> str:
        """
        Generate a text snippet from document with highlighted context.

        Args:
            doc_file: Document filename
            highlight_terms: Terms to find and highlight context around
            max_length: Maximum snippet length

        Returns:
            Text snippet with context around first found term
        """
        doc_path = os.path.join("corpus", doc_file)

        try:
            with open(doc_path, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            return f"[Document {doc_file} not found]"
        except Exception as e:
            return f"[Error reading document: {e}]"

        # Clean up whitespace
        text = ' '.join(text.split())
        lower_text = text.lower()

        # Find first occurrence of any query term
        best_position = None
        for term in highlight_terms:
            pos = lower_text.find(term)
            if pos >= 0:
                if best_position is None or pos < best_position:
                    best_position = pos

        # Generate snippet
        if best_position is None:
            # No term found, return beginning of document
            snippet = text[:min(SNIPPET_FALLBACK_LENGTH, len(text))]
        else:
            # Extract context around the found term
            start = max(0, best_position - SNIPPET_CONTEXT_BEFORE)
            end = min(len(text), best_position + SNIPPET_CONTEXT_AFTER)
            snippet = text[start:end]

            # Add ellipsis if truncated
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."

        return snippet

    def search(self, query: str) -> None:
        """
        Execute a search query and display results.

        Args:
            query: User's search query (keywords or "phrase")
        """
        if not query.strip():
            print("Error: Empty query")
            return

        # Check for phrase search (enclosed in quotes)
        if query.startswith('"') and query.endswith('"') and len(query) > 2:
            self._search_phrase(query[1:-1])
        else:
            self._search_keywords(query)

    def _search_phrase(self, phrase: str) -> None:
        """Execute exact phrase search."""
        phrase_terms = self.tokenize(phrase)

        if not phrase_terms:
            print("No valid terms in phrase.")
            return

        print(f"\nSearching for exact phrase: \"{phrase}\"")
        print("-" * 50)

        matches = self.find_phrase_matches(phrase_terms)

        if not matches:
            print("No documents contain this exact phrase.")
            return

        print(f"Found {len(matches)} document(s) with exact phrase:\n")

        for i, doc in enumerate(matches, 1):
            print(f"{i}. {doc}")
            snippet = self.generate_snippet(doc, phrase_terms)
            print(f"   {snippet}\n")

    def _search_keywords(self, query: str) -> None:
        """Execute keyword search with TF-IDF ranking."""
        terms = self.tokenize(query)

        if not terms:
            print("No valid search terms found.")
            return

        print(f"\nSearching for keywords: {', '.join(terms)}")
        print("-" * 50)

        scores = self.calculate_tf_idf_scores(terms)

        if not scores:
            print("No documents found matching your query.")
            return

        # Sort by score (descending)
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print(f"Found {len(ranked_results)} document(s):\n")

        # Display top results
        for rank, (doc, score) in enumerate(ranked_results[:MAX_RESULTS_DISPLAY], 1):
            print(f"{rank}. {doc} (relevance: {score:.3f})")
            snippet = self.generate_snippet(doc, terms)
            print(f"   {snippet}\n")

        if len(ranked_results) > MAX_RESULTS_DISPLAY:
            print(f"... and {len(ranked_results) - MAX_RESULTS_DISPLAY} more results")

    def get_statistics(self) -> Dict[str, int]:
        """Get index statistics."""
        total_postings = sum(
            len(docs) for docs in self.index.values()
        )
        avg_postings = total_postings / len(self.index) if self.index else 0

        return {
            "num_documents": len(self.documents),
            "num_terms": len(self.index),
            "total_postings": total_postings,
            "avg_postings_per_term": avg_postings
        }


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("\nUsage: python search.py <query>")
        print("\nExamples:")
        print('  python search.py "machine learning"  # Keyword search')
        print('  python search.py \'"exact phrase"\'   # Exact phrase search')
        print("\nOptions:")
        print("  --stats    Show index statistics")
        return

    # Handle special commands
    if sys.argv[1] == "--stats":
        engine = SearchEngine()
        stats = engine.get_statistics()
        print("\nIndex Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value:,.0f}")
        return

    # Execute search
    query = ' '.join(sys.argv[1:])
    engine = SearchEngine()
    engine.search(query)


if __name__ == "__main__":
    main()