from collections import defaultdict
import math
import sklearn.metrics 

class ICMCalculator:
    def __init__(self, num_labels, labels):
        self.num_labels = num_labels
        self.labels = labels
        if len(self.labels) == 0:
            raise ValueError("Labels list cannot be empty.")
        counts = [sum(item[i] == 1 for item in labels) for i in range(num_labels)]
        
        probabilities = [count / len(labels) for count in counts]

        # laplace smoothing
        probabilities = [prob + (1/len(self.labels)) for prob in probabilities]

        self.gbv_parent_idx = num_labels
        self.root_idx = num_labels + 1

        gbv_parent_prob = sum(1 for item in labels if any(item[i] == 1 for i in range(1, num_labels))) / len(labels) + (1/len(self.labels)) 
        self.probabilities = probabilities + [gbv_parent_prob, 1.0] # add probabilities for GBV parent and root nodes
        #print(f"Calculated probabilities for ICM: {self.probabilities}")

    def ic(self,label):
        return - math.log(self.probabilities[label], 2)

    def los(self, label1, label2):
        if label1 == label2:
            return label1
        if 0 in (label1, label2):
            return self.root_idx # if either label is non-GBV, LSO is root
        return self.gbv_parent_idx # if both labels are GBV but different, LSO is GBV parent node
        
    def get_ic(self, item, depth=0):
        positives = [i for i,v in enumerate(item) if v == 1] if depth == 0 else item 

        #print(f"Positives: {positives}")
        if len(positives) == 0:
            return 0.0

        elif len(positives) == 1:
            return self.ic(positives[0])
        
        c1 = positives.pop(0)
        lso_concepts = [self.los(c1, ci) for ci in positives]
        #print(lso_concepts)

        return self.ic(c1) + self.get_ic(positives, depth=1) - self.get_ic(lso_concepts, depth=1)

    def calculate_icm(self, predictions):
        icm = 0
        for gold, pred in zip(self.labels, predictions):
            union = [1 if g + p > 0 else 0 for g, p in zip(gold, pred)]
            icm += 2 * self.get_ic(gold) + 2 * self.get_ic(pred) - 3 * self.get_ic(union)
        return icm / len(self.labels)


def get_f1_score(records):
    f1_score = defaultdict(int)
    y_true_binary = [record["true_binary"] for record in records]
    y_pred_binary = [1 if record["pred_binary_prob"] >= 0.5 else 0 for record in records]
    f1_score["binary"] = sklearn.metrics.f1_score(y_true_binary, y_pred_binary)

    y_true_category = [record["true_category"] for record in records]
    y_pred_category = [record["pred_category"] for record in records]
    f1_score["category"] = sklearn.metrics.f1_score(y_true_category, y_pred_category, average=None)
    f1_score["category_macro"] = sklearn.metrics.f1_score(y_true_category, y_pred_category, average="macro")
    return f1_score

def calculate_icm(gold_labels, pred_labels, num_category_labels=6):
    gold_standard_icm = ICMCalculator(num_category_labels, gold_labels).calculate_icm(gold_labels)
    majority_class_icm = ICMCalculator(num_category_labels, gold_labels).calculate_icm([[1,0,0,0,0,0] for _ in gold_labels]) # all predictions are non-GBV
    minority_class_icm = ICMCalculator(num_category_labels, gold_labels).calculate_icm([[0,0,0,0,1,0] for _ in gold_labels]) # all predictions are the most common GBV category
    predicted_icm = ICMCalculator(num_category_labels, gold_labels).calculate_icm(pred_labels)
    print(f"Gold standard ICM: {gold_standard_icm:.4f}, Majority class ICM: {majority_class_icm:.4f}, Minority class ICM: {minority_class_icm:.4f}, Predicted ICM: {predicted_icm:.4f}")
