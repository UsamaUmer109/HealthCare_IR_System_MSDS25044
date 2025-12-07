"""
TF-IDF Retriever (Final Version with JSON Saving)
Supports token-level + phrase-level indexing and full JSON saving/loading.
"""

import pandas as pd
import numpy as np
import math
import json
import os
import ast
from typing import List, Dict


class TFIDFRetriever:
    def __init__(self, data_path="data/processed/retrieval_dataset.csv"):
        self.data_path = data_path
        self.df = None
        self.index = {}     # inverted index
        self.idf = {}       # idf values
        self.vectors = {}   # tf-idf vectors
        self.N = 0          # number of documents
        self.load_data()

    # ============================================================
    #                     LOAD DATA
    # ============================================================
    def load_data(self):
        print("Loading data for TF-IDF...")

        if not os.path.exists(self.data_path):
            print(f"❌ File not found: {self.data_path}")
            return False

        try:
            self.df = pd.read_csv(self.data_path)

            def parse(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, str) and x.startswith("["):
                    try:
                        return ast.literal_eval(x)
                    except:
                        return []
                return []

            self.df["symptoms"] = self.df["symptoms"].apply(parse)
            self.df["expanded_symptoms"] = self.df["expanded_symptoms"].apply(parse)
            self.df["precautions"] = self.df["precautions"].apply(parse)

            self.N = len(self.df)
            print(f"✅ Loaded {self.N} documents")
            return True

        except Exception as e:
            print("❌ Error loading CSV:", e)
            return False

    # ============================================================
    #                 BUILD TOKENIZED INDEX
    # ============================================================
    def build_inverted_index(self):
        print("\nBuilding TF-IDF inverted index (tokenized)...")

        self.index = {}

        for doc_id, row in self.df.iterrows():
            terms = []

            # Add phrase + token tokens
            for sym in row["expanded_symptoms"]:
                sym = sym.strip().lower()
                if sym:
                    terms.append(sym)
                    for t in sym.split():
                        if len(t) > 2:
                            terms.append(t)

            # Count term frequencies
            counts = {}
            for t in terms:
                counts[t] = counts.get(t, 0) + 1

            # Add to inverted index
            for t, c in counts.items():
                if t not in self.index:
                    self.index[t] = {}
                self.index[t][doc_id] = c

        print(f"✅ Inverted index ready with {len(self.index)} unique terms")
        return self.index

    # ============================================================
    #                       IDF
    # ============================================================
    def calculate_idf(self):
        print("\nCalculating IDF values...")

        self.idf = {}
        for term, docs in self.index.items():
            df = len(docs)
            self.idf[term] = math.log(self.N / (df + 1)) + 1

        print(f"✅ IDF calculated for {len(self.idf)} terms")
        return self.idf

    # ============================================================
    #                    TF-IDF VECTOR BUILD
    # ============================================================
    def build_tfidf_vectors(self):
        print("\nBuilding TF-IDF vectors...")

        self.vectors = {i: {} for i in range(self.N)}

        for term, docs in self.index.items():
            for doc_id, freq in docs.items():
                self.vectors[doc_id][term] = freq * self.idf[term]

        print(f"✅ Built TF-IDF vectors for {len(self.vectors)} documents")
        return self.vectors

    # ============================================================
    #                    MASTER BUILDER
    # ============================================================
    def build_index(self):
        self.build_inverted_index()
        self.calculate_idf()
        self.build_tfidf_vectors()
        return self.index, self.idf, self.vectors

    # ============================================================
    #                      JSON SAVE
    # ============================================================
    def save_index(self, filepath="data/indices/tfidf_index.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "N": self.N,
            "index": self.index,
            "idf": self.idf,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved TF-IDF index to: {filepath}")

    # ============================================================
    #                      JSON LOAD  (FIX APPLIED)
    # ============================================================
    def load_index(self, filepath="data/indices/tfidf_index.json"):
        if not os.path.exists(filepath):
            print(f"❌ No saved TF-IDF index found at: {filepath}")
            return False

        print(f"Loading saved TF-IDF index from: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.N = data["N"]
        raw_index = data["index"]
        self.idf = data["idf"]

        # ---------------------------------------------------------
        # 🔧 FIX: Convert doc_id keys from STRING → INTEGER
        # ---------------------------------------------------------
        fixed_index = {}
        for term, posting_dict in raw_index.items():
            new_postings = {}
            for doc_id_str, freq in posting_dict.items():
                try:
                    new_postings[int(doc_id_str)] = freq
                except:
                    continue
            fixed_index[term] = new_postings

        self.index = fixed_index
        # ---------------------------------------------------------

        # rebuild vectors using corrected index
        self.build_tfidf_vectors()

        print(f"✅ Loaded TF-IDF index with {len(self.index)} terms")
        return True

    # ============================================================
    #                   QUERY PROCESSING
    # ============================================================
    def preprocess_query(self, query):
        words = query.lower().split()
        return [w for w in words if len(w) > 2]

    # ============================================================
    #                      SEARCH ENGINE
    # ============================================================
    def search(self, query, top_k=10):
        query_words = self.preprocess_query(query)

        qvec = {}
        for w in query_words:
            if w in self.idf:
                qvec[w] = self.idf[w]

        if not qvec:
            return []

        results = []

        for doc_id in range(self.N):
            dvec = self.vectors[doc_id]

            dot = 0
            qnorm = 0
            dnorm = 0

            for w, qw in qvec.items():
                qnorm += qw * qw
                if w in dvec:
                    dw = dvec[w]
                    dot += qw * dw
                    dnorm += dw * dw

            if qnorm > 0 and dnorm > 0:
                score = dot / (math.sqrt(qnorm) * math.sqrt(dnorm))
                if score > 0:
                    row = self.df.iloc[doc_id]
                    matched = [w for w in query_words if w in row["expanded_symptoms"]]

                    results.append({
                        "doc_id": doc_id,
                        "disease": row["disease"],
                        "score": score,
                        "matched_terms": matched,
                        "symptoms": row["symptoms"],
                        "precautions": row["precautions"],
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ============================================================
    #                FORMAT RESULTS
    # ============================================================
    def format_results(self, results, query):
        if not results:
            return f"No TF-IDF results for '{query}'"

        out = f"\n🔍 TF-IDF Results for: '{query}'\n" + "="*60 + "\n"

        for i, r in enumerate(results, 1):
            out += f"{i}. {r['disease'].title()}  (Score: {r['score']:.4f})\n"
            out += f"   Matched: {', '.join(r['matched_terms'])}\n"
            out += f"   Symptoms: {', '.join(r['symptoms'][:4])}\n"
            out += "-" * 50 + "\n"

        return out


# ============================================================
#                   TEST HARNESS
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTING TF-IDF RETRIEVER")
    print("=" * 70)

    retriever = TFIDFRetriever()

    # Load saved index, or build & save
    if not retriever.load_index():
        retriever.build_index()
        retriever.save_index()

    print("\n2️⃣  TESTING EXAMPLE QUERIES")

    test_queries = [
        "skin rash",
        "itching skin",
        "abdominal pain",
        "fatigue vomiting",
        "yellow eyes",
        "joint swelling pain",
    ]

    for q in test_queries:
        print("\n--------------------------------------")
        print(f"🔍 Query: {q}")
        results = retriever.search(q)
        print(retriever.format_results(results, q))

    print("\n" + "="*70)
    print("🎮 INTERACTIVE TF-IDF SEARCH")
    print("="*70)

    while True:
        q = input("\nEnter symptoms (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        results = retriever.search(q)
        print(retriever.format_results(results, q))
