# src/evaluation/evaluation.py

import json
import math

class Evaluator:
    def __init__(self, ir_system, eval_file="data/evaluation/evaluation_queries.json"):
        self.ir = ir_system
        self.eval_file = eval_file

        with open(eval_file, "r") as f:
            self.queries = json.load(f)

    # -----------------------------------------
    # Basic Metrics
    # -----------------------------------------
    def precision_at_k(self, retrieved, relevant, k):
        retrieved_set = set(r["disease"].lower() for r in retrieved[:k])
        relevant_set = set(r.lower() for r in relevant)

        tp = len(retrieved_set & relevant_set)
        return tp / k

    def recall_at_k(self, retrieved, relevant, k):
        retrieved_set = set(r["disease"].lower() for r in retrieved[:k])
        relevant_set = set(r.lower() for r in relevant)

        tp = len(retrieved_set & relevant_set)
        return tp / len(relevant_set)

    def f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    # -----------------------------------------
    # MAP
    # -----------------------------------------
    def average_precision(self, retrieved, relevant):
        ap = 0
        hits = 0

        for i, item in enumerate(retrieved, 1):
            if item["disease"].lower() in relevant:
                hits += 1
                ap += hits / i

        if not relevant:
            return 0

        return ap / len(relevant)

    # -----------------------------------------
    # NDCG
    # -----------------------------------------
    def ndcg_at_k(self, retrieved, relevant, k):
        dcg = 0
        for i, item in enumerate(retrieved[:k], 1):
            if item["disease"].lower() in relevant:
                dcg += 1 / math.log2(i + 1)

        ideal_hits = min(k, len(relevant))
        idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

        if idcg == 0:
            return 0

        return dcg / idcg

    # -----------------------------------------
    # Run Evaluation
    # -----------------------------------------
    def evaluate_model(self, model_name, search_fn, k=5):
        results = []

        for query, relevant in self.queries.items():
            retrieved = search_fn(query)

            p = self.precision_at_k(retrieved, relevant, k)
            r = self.recall_at_k(retrieved, relevant, k)
            f1 = self.f1_score(p, r)
            ap = self.average_precision(retrieved, relevant)
            ndcg = self.ndcg_at_k(retrieved, relevant, k)

            results.append((p, r, f1, ap, ndcg))

        avg = lambda x: sum(x) / len(x)
        metrics = {
            "precision": avg([r[0] for r in results]),
            "recall": avg([r[1] for r in results]),
            "f1": avg([r[2] for r in results]),
            "map": avg([r[3] for r in results]),
            "ndcg": avg([r[4] for r in results]),
        }

        print(f"\n📊 RESULTS — {model_name}:")
        print(metrics)
        return metrics
