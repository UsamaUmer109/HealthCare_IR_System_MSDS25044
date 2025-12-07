"""
COMPLETE HEALTHCARE IR SYSTEM WITH ALL RETRIEVAL MODELS
Boolean + TF-IDF + BM25 + Hybrid Fusion
"""

import os
from typing import List, Dict
import numpy as np

# Import all retrievers
from retrieval.boolean_retriever import BooleanRetriever
from retrieval.tfidf_retriever import TFIDFRetriever
from retrieval.bm25_retriever import BM25Retriever
from evidence.evidence_retriever import EvidenceRetriever


class CompleteHealthcareIRSystem:
    def __init__(self):
        self.boolean_ret = None
        self.tfidf_ret = None
        self.bm25_ret = None
        self.initialized = False
        self.evidence_ret = EvidenceRetriever()


    # ======================================================================
    # INITIALIZATION
    # ======================================================================
    def initialize(self) -> bool:
        print("=" * 70)
        print("🏥 COMPLETE HEALTHCARE IR SYSTEM INITIALIZATION")
        print("=" * 70)
        print("Loading all retrieval models...")

        # --------------------- BOOLEAN RETRIEVER ---------------------
        print("\n1️⃣  Loading Boolean Retriever...")
        self.boolean_ret = BooleanRetriever()

        if self.boolean_ret.df is None:
            print("❌ Failed to load Boolean retriever")
            return False

        print("   🔧 Building Boolean index (fresh)...")
        self.boolean_ret.build_index()
        print("   ✅ Boolean: Ready")

        # --------------------- TF-IDF RETRIEVER ----------------------
        print("\n2️⃣  Loading TF-IDF Retriever...")
        self.tfidf_ret = TFIDFRetriever()

        if self.tfidf_ret.df is None:
            print("❌ Failed to load TF-IDF retriever")
            return False

        tfidf_path = "data/indices/tfidf_index.json"

        if os.path.exists(tfidf_path):
            self.tfidf_ret.load_index(tfidf_path)

            # 🔧 PATCH: FIX STRING DOC IDS → INT (prevents KeyError: '0')
            fixed_index = {}
            for term, postings in self.tfidf_ret.index.items():
                new_postings = {}
                for doc_id_str, freq in postings.items():
                    new_postings[int(doc_id_str)] = freq
                fixed_index[term] = new_postings

            self.tfidf_ret.index = fixed_index

            # rebuild vectors because index was patched
            self.tfidf_ret.build_tfidf_vectors()

        else:
            print("   🔧 Building TF-IDF index...")
            self.tfidf_ret.build_index()
            self.tfidf_ret.save_index(tfidf_path)

        print("   ✅ TF-IDF: Ready")

        # ---------------------- BM25 RETRIEVER -----------------------
        print("\n3️⃣  Loading BM25 Retriever...")
        self.bm25_ret = BM25Retriever()

        if self.bm25_ret.df is None:
            print("❌ Failed to load BM25 retriever")
            return False

        bm25_path = "data/indices/bm25_index.json"

        if os.path.exists(bm25_path):
            self.bm25_ret.load_index(bm25_path)
        else:
            print("   🔧 Building BM25 index...")
            self.bm25_ret.build_index()
            self.bm25_ret.save_index(bm25_path)

        print("   ✅ BM25: Ready")
        print("\n4️⃣  Loading Evidence Retriever...")
        self.evidence_ret = EvidenceRetriever()

        if self.evidence_ret.df is None:
            print("❌ Failed to load Evidence retriever")
            return False

        print("   ✅ Evidence: Ready")
    
        # ---------------------- DONE -----------------------
        self.initialized = True

        print("\n" + "=" * 70)
        print("✅ ALL RETRIEVAL MODELS INITIALIZED SUCCESSFULLY!")
        print("=" * 70)

        self._print_system_stats()

        return True

    # ======================================================================
    # SYSTEM STATS
    # ======================================================================
    def _print_system_stats(self):
        print("\n📊 SYSTEM STATISTICS:")
        print(f"   • Total documents: {len(self.boolean_ret.df)}")
        print(f"   • Boolean index terms: {len(self.boolean_ret.inverted_index)}")
        print(f"   • TF-IDF index terms: {len(self.tfidf_ret.index)}")
        print(f"   • BM25 index terms: {len(self.bm25_ret.index)}")

        # precautions count
        def safe_len(x):
            if isinstance(x, list):
                return len(x)
            try:
                val = eval(x)
                return len(val) if isinstance(val, list) else 0
            except:
                return 0

        print(f"   • Diseases with precautions: {self.boolean_ret.df['precautions'].apply(safe_len).gt(0).sum()}")

        print("\n📝 SAMPLE DISEASES:")
        for i, disease in enumerate(self.boolean_ret.df['disease'].unique()[:5], 1):
            print(f"   {i}. {disease.title()}")

    # ======================================================================
    # WRAPPER METHODS
    # ======================================================================
    def boolean_search(self, query: str):
        return self.boolean_ret.search(query, operator="AND") if self.initialized else []

    def tfidf_search(self, query: str, top_k=10):
        return self.tfidf_ret.search(query, top_k) if self.initialized else []

    def bm25_search(self, query: str, top_k=10):
        return self.bm25_ret.search(query, top_k) if self.initialized else []

    # ======================================================================
    # HYBRID FUSION
    # ======================================================================
    def hybrid_fusion_search(self, query: str, top_k=10, weights=None):
        if not self.initialized:
            return []

        if weights is None:
            weights = {"boolean": 0.2, "tfidf": 0.3, "bm25": 0.5}

        print(f"\n🔍 Performing hybrid fusion search for: '{query}'")
        bool_r = self.boolean_search(query)
        tfidf_r = self.tfidf_search(query, 20)
        bm25_r = self.bm25_search(query, 20)

        # normalize scores
        def normalize(results, k="score"):
            if not results:
                return {}
            scores = [r[k] for r in results]
            lo, hi = min(scores), max(scores)
            if hi == lo:
                return {int(r["doc_id"]): 1.0 for r in results}
            return {int(r["doc_id"]): (r[k] - lo) / (hi - lo) for r in results}

        bn = normalize(bool_r, "score")
        tn = normalize(tfidf_r, "score")
        mn = normalize(bm25_r, "score")

        all_ids = set(bn) | set(tn) | set(mn)
        combined = {}

        for doc_id in all_ids:
            score = weights["boolean"] * bn.get(doc_id, 0) + \
                    weights["tfidf"] * tn.get(doc_id, 0) + \
                    weights["bm25"] * mn.get(doc_id, 0)

            row = self.boolean_ret.df.iloc[doc_id]
            precautions = row["precautions"]
            if isinstance(precautions, str):
                try:
                    precautions = eval(precautions)
                except:
                    precautions = []

            combined[doc_id] = {
                "doc_id": doc_id,
                "disease": row["disease"],
                "combined_score": score,
                "symptoms": row["symptoms"],
                "precautions": precautions,
                "match_types": [
                    m for m, v in
                    {"Boolean": bn.get(doc_id, 0),
                     "TF-IDF": tn.get(doc_id, 0),
                     "BM25": mn.get(doc_id, 0)}.items()
                    if v > 0
                ],
                "individual_scores": {
                    "boolean": bn.get(doc_id, 0),
                    "tfidf": tn.get(doc_id, 0),
                    "bm25": mn.get(doc_id, 0),
                },
            }

        final = list(combined.values())
        final.sort(key=lambda x: x["combined_score"], reverse=True)
        return final[:top_k]

    # ======================================================================
    # COMPARISON VIEW
    # ======================================================================
    def compare_all_methods(self, query: str):
        print(f"\n📊 COMPARING METHODS FOR: {query}")
        print("=" * 80)

        for name, res in [
            ("Boolean", self.boolean_search(query)),
            ("TF-IDF", self.tfidf_search(query)),
            ("BM25", self.bm25_search(query)),
            ("Hybrid", self.hybrid_fusion_search(query)),
        ]:
            print(f"\n{name}: {len(res)} results")
            for r in res[:3]:
                score = r.get("combined_score", r.get("score", 0))
                print(f"   - {r['disease']} ({score:.4f})")

    # ======================================================================
    # RESULT FORMATTER
    # ======================================================================
    def format_results(self, results, query, method="Hybrid"):
        if not results:
            return f"\n❌ No results for: '{query}'\n"

        out = "\n" + "=" * 80
        out += f"\n🏥 {method.upper()} SEARCH RESULTS"
        out += "\n" + "=" * 80
        out += f"\n🔍 Query: {query}"
        out += f"\n📊 {len(results)} diseases found"
        out += "\n" + "=" * 80 + "\n"

        for i, r in enumerate(results[:10], 1):
            out += f"\n{i}. {r['disease'].title()}\n"

            if method.lower() == "hybrid":
                out += f"   ⭐ Combined Score: {r['combined_score']:.2f}\n"
                out += f"   📈 Methods: {', '.join(r['match_types'])}\n"

            out += f"   🏥 Symptoms: {', '.join(r['symptoms'][:4])}\n"

            if r["precautions"]:
                out += f"   💡 Precaution: {r['precautions'][0]}\n"

            out += "-" * 60 + "\n"
            # ================================
            # 🔍 Add Evidence Retrieval
            # ================================
            if hasattr(self, "evidence_ret") and self.evidence_ret.df is not None:
                evidence = self.evidence_ret.search_evidence(r["disease"], top_k=2)

                if evidence:
                    out += f"   📚 Evidence Articles:\n"
                    for ev in evidence:
                        out += f"      • {ev['title']} — {ev['summary'][:120]}...\n"

        return out


# ======================================================================
# MAIN PROGRAM LOOP
# ======================================================================
def main():
    print("=" * 80)
    print("🏥 HEALTHCARE INFORMATION RETRIEVAL SYSTEM")
    print("Boolean + TF-IDF + BM25 + Hybrid Fusion")
    print("=" * 80)

    ir = CompleteHealthcareIRSystem()
    if not ir.initialize():
        print("❌ System failed to initialize.")
        return

    print("\n🎮 INTERACTIVE SEARCH INTERFACE")
    print("=" * 80)
    print("Commands:")
    print("  symptoms text       → Hybrid search")
    print("  boolean text        → Boolean only")
    print("  tfidf text          → TF-IDF only")
    print("  bm25 text           → BM25 only")
    print("  compare text        → Compare all")
    print("  stats               → System stats")
    print("  quit                → Exit")
    print("=" * 80)

    while True:
        try:
            cmd = input("\n🔍 Enter command: ").strip()

            if not cmd:
                continue

            lower = cmd.lower()

            if lower == "quit":
                print("\n👋 Goodbye!")
                break

            if lower == "stats":
                ir._print_system_stats()
                continue

            if lower.startswith("compare "):
                query = cmd[8:].strip()
                ir.compare_all_methods(query)
                continue

            if lower.startswith("boolean "):
                query = cmd[8:].strip()
                res = ir.boolean_search(query)
                print(ir.format_results(res, query, "Boolean"))
                continue

            if lower.startswith("tfidf "):
                query = cmd[6:].strip()
                res = ir.tfidf_search(query)
                print(ir.format_results(res, query, "TF-IDF"))
                continue

            if lower.startswith("bm25 "):
                query = cmd[5:].strip()
                res = ir.bm25_search(query)
                print(ir.format_results(res, query, "BM25"))
                continue

            # Default → Hybrid
            res = ir.hybrid_fusion_search(cmd)
            print(ir.format_results(res, cmd, "Hybrid"))

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
