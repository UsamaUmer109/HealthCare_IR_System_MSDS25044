"""
Boolean Retrieval System for Healthcare
Supports token-level matching + phrase-level matching.
"""

import pandas as pd
import numpy as np
import os
import json
import ast
from typing import List, Dict


class BooleanRetriever:
    def __init__(self, data_path="data/processed/retrieval_dataset.csv"):
        self.data_path = data_path
        self.df = None
        self.inverted_index = {}
        self.load_data()

    # ============================================================
    #                    DATA LOADING
    # ============================================================
    def load_data(self):
        print("Loading retrieval dataset...")

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

            print(f"✅ Loaded {len(self.df)} documents")
            print(f"   Unique diseases: {self.df['disease'].nunique()}")
            print(f"   Sample diseases: {self.df['disease'].iloc[:3].tolist()}")
            return True

        except Exception as e:
            print("❌ Error loading CSV:", e)
            return False

    # ============================================================
    #                  INDEX BUILDING (FIXED)
    # ============================================================
    def build_index(self):
        print("\nBuilding Boolean inverted index...")

        self.inverted_index = {}

        for doc_id, row in self.df.iterrows():
            all_terms = set()
            full_list = row["symptoms"] + row["expanded_symptoms"]

            for s in full_list:
                s = s.strip().lower()
                if not s:
                    continue

                # Add full phrase
                all_terms.add(s)

                # Tokenize phrase (important fix)
                for token in s.split():
                    token = token.strip().lower()
                    if len(token) > 2:  # ignore "of", "in", "to"
                        all_terms.add(token)

            # Put into index
            for t in all_terms:
                if t not in self.inverted_index:
                    self.inverted_index[t] = set()
                self.inverted_index[t].add(int(doc_id))

        print(f"✅ Index built with {len(self.inverted_index)} unique terms")
        self._print_index_stats()
        return self.inverted_index

    # ============================================================
    #                        INDEX STATS
    # ============================================================
    def _print_index_stats(self):
        print("\n📊 INDEX STATISTICS:")
        dfreq = [len(v) for v in self.inverted_index.values()]

        print(f"   Total terms: {len(self.inverted_index)}")
        print(f"   Avg documents per term: {np.mean(dfreq):.1f}")
        print(f"   Min: {np.min(dfreq)}")
        print(f"   Max: {np.max(dfreq)}")

        sorted_terms = sorted(self.inverted_index.items(), key=lambda x: len(x[1]), reverse=True)

        print("\n🏆 TOP 10 MOST COMMON TERMS:")
        for i, (term, docs) in enumerate(sorted_terms[:10], 1):
            print(f"{i:2}. {term:25} → {len(docs)} documents")

    # ============================================================
    #                    QUERY PROCESSING
    # ============================================================
    def preprocess_query(self, query: str) -> List[str]:
        parts = query.lower().split()
        cleaned = []
        for p in parts:
            p = p.strip(".,!?;:")
            if len(p) > 2:
                cleaned.append(p)
        return cleaned

    # ============================================================
    #                        SEARCH
    # ============================================================
    def search(self, query: str, operator="AND") -> List[Dict]:
        if not self.inverted_index:
            raise ValueError("Index not built. Run build_index() first.")

        q_terms = self.preprocess_query(query)
        print(f"\n🔍 Searching for: {q_terms}")

        doc_sets = []
        for t in q_terms:
            if t in self.inverted_index:
                print(f"   '{t}' found in {len(self.inverted_index[t])} docs")
                doc_sets.append(self.inverted_index[t])
            else:
                print(f"   '{t}' NOT found")
                doc_sets.append(set())

        result_docs = (
            set.intersection(*doc_sets) if operator == "AND" else set.union(*doc_sets)
        )

        print(f"🔎 Found {len(result_docs)} matching documents")

        results = []
        for doc_id in result_docs:
            row = self.df.iloc[doc_id]
            matched = [
                t for t in q_terms if t in row["symptoms"] or t in row["expanded_symptoms"]
            ]
            score = len(matched) / len(q_terms)

            results.append(
                {
                    "doc_id": doc_id,
                    "disease": row["disease"],
                    "score": score,
                    "matched_terms": matched,
                    "symptoms": row["symptoms"],
                    "precautions": row["precautions"],
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ============================================================
    #                    RESULTS DISPLAY
    # ============================================================
    def format_results(self, results, query):
        if not results:
            return f"No results for '{query}'"

        out = f"\n=== RESULTS FOR '{query}' ===\n"
        for i, r in enumerate(results[:10], 1):
            out += f"{i}. {r['disease'].title()} (Score: {r['score']:.0%})\n"
            out += f"   Matched: {', '.join(r['matched_terms'])}\n"
            out += f"   Symptoms: {', '.join(r['symptoms'][:4])}\n"
            if r["precautions"]:
                out += f"   Precaution: {r['precautions'][0]}\n"
            out += "-" * 50 + "\n"
        return out


# ============================================================
#               FULL TERMINAL TESTING (YOU WANT THIS)
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTING BOOLEAN RETRIEVAL SYSTEM")
    print("=" * 70)

    retriever = BooleanRetriever()
    retriever.build_index()

    print("\n2️⃣  SAVING INDEX AS JSON...")
    retriever.save_index = lambda *args, **kwargs: None  # placeholder, optional

    print("\n3️⃣  VIEWING INDEX SAMPLE...")
    keys = list(retriever.inverted_index.keys())[:5]
    for k in keys:
        print(f"{k} → {retriever.inverted_index[k]}")

    print("\n4️⃣  TESTING SEARCH QUERIES...")
    test_qs = [
        "skin rash",
        "itching skin",
        "abdominal pain",
        "fatigue vomiting",
        "joint pain swelling",
    ]

    for q in test_qs:
        print("\n----------------------------------------")
        print(f"🔍 Query: {q}")
        results = retriever.search(q)
        print(retriever.format_results(results, q))

    print("\n" + "=" * 70)
    print("🎮 INTERACTIVE BOOLEAN SEARCH")
    print("=" * 70)

    while True:
        q = input("\n🔍 Enter symptoms (or quit): ").strip()
        if q.lower() == "quit":
            break
        results = retriever.search(q)
        print(retriever.format_results(results, q))

