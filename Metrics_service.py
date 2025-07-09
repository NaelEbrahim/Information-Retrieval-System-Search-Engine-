from sklearn.metrics import average_precision_score
import numpy as np


class RetrievalEvaluator:
    def __init__(self, successful_query, all_query, k=10):
        self.k = k
        self.successful_query = successful_query
        self.all_query = all_query

    def precision_at_k(self, y_true, y_pred):
        y_true_set = set(y_true)
        y_pred_at_k = y_pred[:self.k]
        return len(set(y_pred_at_k) & y_true_set) / self.k

    def recall_at_k(self, y_true, y_pred):
        y_true_set = set(y_true)
        y_pred_at_k = y_pred[:self.k]
        return len(set(y_pred_at_k) & y_true_set) / len(y_true_set) if y_true_set else 0

    def mean_average_precision(self, all_y_true, all_y_pred):
        ap_scores = []
        for y_true, y_pred in zip(all_y_true, all_y_pred):
            y_true_set = set(y_true)
            score = 0.0
            hits = 0.0
            for i, p in enumerate(y_pred):
                if p in y_true_set:
                    hits += 1.0
                    score += hits / (i + 1.0)
            if hits > 0:
                ap_scores.append(score / len(y_true_set))
            else:
                ap_scores.append(0.0)
        return np.mean(ap_scores)

    def mean_reciprocal_rank(self, all_y_true, all_y_pred):
        rr_scores = []
        for y_true, y_pred in zip(all_y_true, all_y_pred):
            y_true_set = set(y_true)
            for rank, p in enumerate(y_pred, start=1):
                if p in y_true_set:
                    rr_scores.append(1.0 / rank)
                    break
            else:
                rr_scores.append(0.0)
        return np.mean(rr_scores)

    def evaluate(self, all_y_true, all_y_pred):
        return {
            'MAP': self.mean_average_precision(all_y_true, all_y_pred),
            'MRR': self.mean_reciprocal_rank(all_y_true, all_y_pred),
            f'Precision@{self.k}': np.mean([
                self.precision_at_k(y_t, y_p) for y_t, y_p in zip(all_y_true, all_y_pred)
            ]),
            'Recall': np.mean([
                self.recall_at_k(y_t, y_p) for y_t, y_p in zip(all_y_true, all_y_pred)
            ]),
            'successful_query' : self.successful_query,
            'all tested query' : self.all_query
        }
