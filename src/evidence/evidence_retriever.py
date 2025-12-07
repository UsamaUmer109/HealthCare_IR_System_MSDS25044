# src/evidence/evidence_retriever.py

import math
import pandas as pd
import os
import re


class EvidenceRetriever:
    """
    BM25 Evidence Retrieval for diseases using PubMed-style abstracts.
    Includes safe checks if dataset is missing.
    """

    def __init__(self, data_path="data/evidence/pubmed_clean.csv"):
        self.data_path = data_path
        self.df = None
        self.index = {}
        self.idf = {}
        self.avgdl = 0
        self.k1 = 1.5
        self.b = 0.75

        if self.load_data():
            self.build_index()
        else:
            print("⚠️ Evidence retrieval disabled — dataset not found.")

    # --------------------------------------------------------------
    # LOAD RAW PUBMED DATA
    # --------------------------------------------------------------
    def load_data(self):
        if not os.path.exists(self.data_path):
            print(f"❌ Evidence dataset not found: {self.data_path}")
            return False

        try:
            self.df = pd.read_csv(self.data_path)

            self.df["title"] = self.df["title"].fillna("")
            self.df["abstract"] = self.df["abstract"].fillna("")
            self.df["full_text"] = (self.df["title"] + " " + self.df["abstract"]).str.lower()

            print(f"✅ Loaded evidence dataset with {len(self.df)} articles")
            return True

        except Exception as e:
            print("❌ Error loading evidence dataset:", e)
            return False

    # --------------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------------
    def tokenize(self, text):
        return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    # --------------------------------------------------------------
    # BUILD BM25 INDEX
    # --------------------------------------------------------------
    def build_index(self):
        if self.df is None:
            return

        print("🔧 Building BM25 evidence index...")

        self.index = {}
        doc_lengths = []

        for doc_id, row in self.df.iterrows():
            terms = self.tokenize(row["full_text"])
            doc_lengths.append(len(terms))

            freq = {}
            for t in terms:
                freq[t] = freq.get(t, 0) + 1

            for t, c in freq.items():
                if t not in self.index:
                    self.index[t] = {}
                self.index[t][doc_id] = c

        self.avgdl = sum(doc_lengths) / len(doc_lengths)
        N = len(self.df)

        # Compute IDF
        for term, posting in self.index.items():
            df = len(posting)
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        print(f"✅ Evidence index built with {len(self.index)} indexed terms.")

    # --------------------------------------------------------------
    # BM25 SCORING
    # --------------------------------------------------------------
    def bm25_score(self, query_terms, doc_id):
        doc = self.df.iloc[doc_id]
        tokens = self.tokenize(doc["full_text"])
        doc_len = len(tokens)

        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        score = 0
        for term in query_terms:
            if term not in freq or term not in self.idf:
                continue

            f = freq[term]
            idf = self.idf[term]

            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)

        return score

    # --------------------------------------------------------------
    # SEARCH TOP-K EVIDENCE
    # --------------------------------------------------------------
    def search_evidence(self, disease, top_k=3):

        if self.df is None:
            print("⚠️ Cannot search evidence: dataset not loaded.")
            return []

        disease_tokens = self.tokenize(disease)

        scores = []
        for doc_id in range(len(self.df)):
            s = self.bm25_score(disease_tokens, doc_id)
            if s > 0:
                scores.append((doc_id, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = scores[:top_k]

        results = []
        for doc_id, score in top_docs:
            row = self.df.iloc[doc_id]
            summary = self.generate_summary(row["abstract"])

            results.append({
                "doc_id": doc_id,
                "title": row["title"],
                "abstract": row["abstract"],
                "summary": summary,
                "score": score,
            })

        return results

    # --------------------------------------------------------------
    # SIMPLE SUMMARIZER (NO external libraries)
    # --------------------------------------------------------------
    def generate_summary(self, text):
        text = text.strip()
        if not text:
            return ""

        sentences = [s.strip() for s in re.split(r"\.|\?|!", text) if len(s.strip()) > 0]

        if len(sentences) < 3:
            return text

        ranked = sorted(sentences, key=lambda s: len(s), reverse=True)
        top = ranked[:2]

        return ". ".join(top) + "."


# --------------------------------------------------------------
# STANDALONE TEST
# --------------------------------------------------------------
if __name__ == "__main__":
    er = EvidenceRetriever()

    disease = "diabetes"
    print(f"\n🔎 Searching evidence for: {disease}")

    results = er.search_evidence(disease, top_k=3)

    for r in results:
        print("\n--------------------------------")
        print("Title:", r["title"])
        print("Score:", round(r["score"], 4))
        print("Summary:", r["summary"])
