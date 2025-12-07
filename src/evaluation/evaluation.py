import math

class EvaluationMetrics:

    # ----------------------------------------------------------
    # PRECISION@K
    # ----------------------------------------------------------
    def precision_at_k(self, retrieved_ids, relevant_ids, k=5):
        if k == 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        rel = len(set(retrieved_k) & set(relevant_ids))
        return rel / k

    # ----------------------------------------------------------
    # RECALL@K
    # ----------------------------------------------------------
    def recall_at_k(self, retrieved_ids, relevant_ids, k=5):
        if len(relevant_ids) == 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        rel = len(set(retrieved_k) & set(relevant_ids))
        return rel / len(relevant_ids)

    # ----------------------------------------------------------
    # F1@K
    # ----------------------------------------------------------
    def f1_at_k(self, retrieved_ids, relevant_ids, k=5):
        p = self.precision_at_k(retrieved_ids, relevant_ids, k)
        r = self.recall_at_k(retrieved_ids, relevant_ids, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    # ----------------------------------------------------------
    # DCG@K
    # ----------------------------------------------------------
    def dcg_at_k(self, retrieved_ids, relevant_ids, k=5):
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                dcg += 1 / math.log2(i + 2)
        return dcg

    # ----------------------------------------------------------
    # IDCG@K (Ideal DCG)
    # ----------------------------------------------------------
    def idcg_at_k(self, relevant_ids, k=5):
        ideal_rel = min(len(relevant_ids), k)
        idcg = 0.0
        for i in range(ideal_rel):
            idcg += 1 / math.log2(i + 2)
        return idcg

    # ----------------------------------------------------------
    # nDCG@K
    # ----------------------------------------------------------
    def ndcg_at_k(self, retrieved_ids, relevant_ids, k=5):
        idcg = self.idcg_at_k(relevant_ids, k)
        if idcg == 0:
            return 0.0
        dcg = self.dcg_at_k(retrieved_ids, relevant_ids, k)
        return dcg / idcg

    # ----------------------------------------------------------
    # AVERAGE PRECISION
    # ----------------------------------------------------------
    def average_precision(self, retrieved_ids, relevant_ids):
        hits = 0
        sum_precisions = 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                hits += 1
                sum_precisions += hits / (i + 1)

        if hits == 0:
            return 0.0

        return sum_precisions / len(relevant_ids)

    # ----------------------------------------------------------
    # MEAN AVERAGE PRECISION
    # ----------------------------------------------------------
    def mean_average_precision(self, ap_scores):
        if len(ap_scores) == 0:
            return 0.0
        return sum(ap_scores) / len(ap_scores)

