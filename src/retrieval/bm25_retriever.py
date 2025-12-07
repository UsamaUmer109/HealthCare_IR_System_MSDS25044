# src/retrieval/bm25_retriever.py
"""
BM25 Retrieval System for Healthcare
Supports exact term, partial term, and full-phrase matching.
"""

import pandas as pd
import numpy as np
import math
import json
import os
from collections import defaultdict
from typing import List, Dict


class BM25Retriever:
    def __init__(self, data_path="data/processed/retrieval_dataset.csv"):
        self.data_path = data_path

        self.df = None
        self.index = {}           # term -> {doc_id: freq}
        self.doc_lengths = {}     # doc_id -> length
        self.avg_doc_length = 0
        self.doc_freq = {}        # term -> document frequency
        self.N = 0                # total docs

        self.k1 = 1.5
        self.b = 0.75

        self.load_data()

    # ---------------------- LOAD DATA ----------------------
    def load_data(self):
        print("Loading data for BM25...")

        if not os.path.exists(self.data_path):
            print(f"❌ Data file not found: {self.data_path}")
            return False

        try:
            self.df = pd.read_csv(self.data_path)

            # Convert list strings to lists
            self.df["symptoms"] = self.df["symptoms"].apply(
                lambda x: eval(x) if isinstance(x, str) else []
            )
            self.df["expanded_symptoms"] = self.df["expanded_symptoms"].apply(
                lambda x: eval(x) if isinstance(x, str) else []
            )

            self.N = len(self.df)
            print(f"✅ Loaded {self.N} documents for BM25")
            print(f"   Sample diseases: {self.df['disease'].iloc[:3].tolist()}")

            return True

        except Exception as e:
            print("❌ Error while loading data:", e)
            return False

    # ---------------------- BUILD INDEX ----------------------
    def build_index(self):
        print("\nBuilding BM25 index...")

        self.index = {}
        self.doc_lengths = {}
        total_len = 0

        for doc_id, row in self.df.iterrows():
            terms = row["expanded_symptoms"]
            doc_len = len(terms)
            self.doc_lengths[doc_id] = doc_len
            total_len += doc_len

            freq_map = defaultdict(int)
            for t in terms:
                freq_map[t] += 1

            for t, f in freq_map.items():
                if t not in self.index:
                    self.index[t] = {}
                self.index[t][doc_id] = f

        self.avg_doc_length = total_len / self.N
        self.doc_freq = {t: len(docs) for t, docs in self.index.items()}

        print(f"   • Unique terms: {len(self.index)}")
        print(f"   • Avg doc length: {self.avg_doc_length:.2f}")
        print(f"   • Total docs: {self.N}")

        return True

    # ---------------------- IDF ----------------------
    def calculate_idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.N - df + 0.5) / (df + 0.5))

    # ---------------------- BM25 TERM SCORE ----------------------
    def calculate_bm25_score(self, doc_id: int, term: str, freq: int) -> float:
        idf = self.calculate_idf(term)
        doc_len = self.doc_lengths.get(doc_id, 0)

        numerator = freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))

        return idf * (numerator / denominator)

    # ---------------------- QUERY PREPROCESS ----------------------
    def preprocess_query(self, query: str) -> List[str]:
        tokens = query.lower().split()
        return [t for t in tokens if len(t) > 2]

    # ---------------------- SEARCH ----------------------
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_terms = self.preprocess_query(query)
        phrase = query.lower().strip()

        all_query_terms = query_terms + [phrase]  # include phrase search

        scores = defaultdict(float)

        for index_term in self.index.keys():
            for q in all_query_terms:

                # exact match
                if q == index_term:
                    for doc_id, freq in self.index[index_term].items():
                        scores[int(doc_id)] += self.calculate_bm25_score(int(doc_id), index_term, freq)

                # partial match
                elif q in index_term:
                    for doc_id, freq in self.index[index_term].items():
                        scores[int(doc_id)] += self.calculate_bm25_score(int(doc_id), index_term, freq)

        # sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc_id = int(doc_id)

            row = self.df.iloc[doc_id]

            matches = []
            for q in all_query_terms:
                for s in row["expanded_symptoms"]:
                    if q in s:
                        matches.append(q)
                        break

            results.append({
                "doc_id": doc_id,
                "disease": row["disease"],
                "score": float(score),
                "symptoms": row["symptoms"],
                "precautions": eval(row["precautions"]) if isinstance(row["precautions"], str) else [],
                "matched_terms": list(set(matches)),
            })

        return results

    # ---------------------- FORMAT RESULTS ----------------------
    def format_results(self, results: List[Dict], query: str) -> str:
        if not results:
            return f"No BM25 results for: '{query}'"

        out = f"\n🔍 BM25 Results for: '{query}'\n" + "=" * 70 + "\n"

        for i, r in enumerate(results[:5], 1):
            out += f"{i}. {r['disease'].title()} (Score: {r['score']:.4f})\n"
            if r["matched_terms"]:
                out += f"   Matched: {', '.join(r['matched_terms'])}\n"
            out += f"   Symptoms: {', '.join(r['symptoms'][:3])}...\n"
            out += "-" * 60 + "\n"

        return out

    # ---------------------- SAVE INDEX ----------------------
    def save_index(self, filepath="data/indices/bm25_index.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        payload = {
            "index": self.index,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "doc_freq": self.doc_freq,
            "N": self.N,
            "k1": self.k1,
            "b": self.b,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved BM25 index → {filepath}")

    # ---------------------- LOAD INDEX ----------------------
    def load_index(self, filepath="data/indices/bm25_index.json"):
        if not os.path.exists(filepath):
            print("❌ No BM25 index found.")
            return False

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.index = data["index"]
        self.doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
        self.avg_doc_length = data["avg_doc_length"]
        self.doc_freq = data["doc_freq"]
        self.N = data["N"]
        self.k1 = data["k1"]
        self.b = data["b"]

        print(f"✅ Loaded BM25 index with {len(self.index)} terms")
        return True


# ---------------------- INTERACTIVE TEST ----------------------
def test_bm25_retriever():
    print("=" * 70)
    print("🧪 BM25 RETRIEVAL SYSTEM — INTERACTIVE MODE")
    print("=" * 70)

    R = BM25Retriever()

    # Load or build index
    index_path = "data/indices/bm25_index.json"
    if os.path.exists(index_path):
        R.load_index(index_path)
    else:
        R.build_index()
        R.save_index(index_path)

    print("\n🎮 INTERACTIVE BM25 SEARCH")
    print("Type symptoms to search (or 'quit')")
    print("-" * 70)

    while True:
        q = input("\n🔍 Enter symptoms: ").strip()
        if q.lower() == "quit":
            break

        results = R.search(q)
        print(R.format_results(results, q))


if __name__ == "__main__":
    test_bm25_retriever()
