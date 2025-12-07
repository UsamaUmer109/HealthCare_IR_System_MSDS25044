import os
import sys

# Fix Python path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from complete_healthcare_ir_system import CompleteHealthcareIRSystem
from evaluation import EvaluationMetrics


# ----------------------------------------------------------
# Initialize IR System
# ----------------------------------------------------------
ir = CompleteHealthcareIRSystem()
ir.initialize()

eval = EvaluationMetrics()

print("\n=======================================")
print("   INTERACTIVE RETRIEVAL EVALUATION")
print("=======================================\n")
print("Instructions:")
print(" • Type a query (e.g., skin rash)")
print(" • Enter relevance doc_ids as comma separated list (e.g., 0,4,7)")
print(" • Type 'quit' to exit\n")


ap_scores = []  # store AP for MAP calculation

while True:
    query = input("\n🔍 Enter query (or 'quit'): ").strip()
    if query.lower() == "quit":
        break
    
    if not query:
        print("❌ Query cannot be empty.")
        continue

    # Ask user for relevance judgments
    rel_input = input("✔ Enter relevant doc_ids (comma separated): ").strip()
    if not rel_input:
        print("❌ Must provide at least one relevant id.")
        continue

    try:
        relevant_ids = [int(x) for x in rel_input.split(",") if x.strip().isdigit()]
    except:
        print("❌ Invalid format. Use comma separated integers.")
        continue

    # ----------------------------------------------------------
    # RUN RETRIEVAL
    # ----------------------------------------------------------
    results = ir.hybrid_fusion_search(query, top_k=10)
    retrieved_ids = [r["doc_id"] for r in results]

    # ----------------------------------------------------------
    # CALCULATE METRICS
    # ----------------------------------------------------------
    p5 = eval.precision_at_k(retrieved_ids, relevant_ids, k=5)
    r5 = eval.recall_at_k(retrieved_ids, relevant_ids, k=5)
    f1 = eval.f1_at_k(retrieved_ids, relevant_ids, k=5)
    ndcg = eval.ndcg_at_k(retrieved_ids, relevant_ids, k=5)
    ap = eval.average_precision(retrieved_ids, relevant_ids)

    ap_scores.append(ap)

    # ----------------------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------------------
    print("\n============== EVALUATION RESULTS ==============")
    print(f"Query: {query}")
    print(f"Relevant IDs: {relevant_ids}")
    print(f"Retrieved IDs: {retrieved_ids[:5]}")
    print("-----------------------------------------------")
    print(f"Precision@5       : {p5:.3f}")
    print(f"Recall@5          : {r5:.3f}")
    print(f"F1-score@5        : {f1:.3f}")
    print(f"nDCG@5            : {ndcg:.3f}")
    print(f"Average Precision : {ap:.3f}")
    print("================================================\n")


# ----------------------------------------------------------
# FINAL MAP CALCULATION
# ----------------------------------------------------------
if ap_scores:
    MAP = eval.mean_average_precision(ap_scores)
    print("\n=======================================")
    print(f" FINAL MAP (Mean Average Precision): {MAP:.3f}")
    print("=======================================\n")
else:
    print("⚠ No queries evaluated. MAP not available.")
