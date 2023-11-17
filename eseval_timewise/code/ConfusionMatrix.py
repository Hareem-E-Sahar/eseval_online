import sys,csv,os

class Result:
    def __init__(self, commit, actual, predicted):
        self.commit = commit
        self.actual = actual
        self.predicted = predicted

class ConfusionMatrix:
    def __init__(self, result):
        self.result = result
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        #self.compute_metrics()

    def compute_metrics(self):
        for querydoc in self.result:
            if str(querydoc.actual) == 'True' and querydoc.predicted == 'True':
                self.tp+=1
            elif str(querydoc.actual) == 'False' and querydoc.predicted == 'False':
                self.tn+=1
            elif str(querydoc.actual) == 'True' and querydoc.predicted == 'False':
                self.fn+=1
            elif str(querydoc.actual) == 'False' and querydoc.predicted == 'True':
                self.fp+=1

    def accuracy(self):
        total_samples = len(self.result)
        correct_predictions = self.tp + self.tn
        return correct_predictions / total_samples

    def precision(self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def recall_clean_class(self): #recall_class_0 = specificity
        if self.tn + self.fp == 0:
            return 0
        return self.tn / (self.tn + self.fp)

    def recall_buggy_class(self): #recall_class_1 = sensitivity
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def specificity(self):
        return self.recall_clean_class()

    def sensitivity(self):
        return self.recall_buggy_class()

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def g_mean(self):
        return (self.sensitivity() * self.specificity()) ** 0.5

    def save_result_list(self,file_path):
        with open(file_path,'w') as file:
            writer = csv.writer(file)
            header = ["commit_id","actual","predicted"]
            writer.writerow(header)
            for doc in self.result:
                row = [doc.commit,doc.actual,doc.predicted]
                writer.writerow(row)

    def display_to_file(self, file_path):
        with open(file_path, 'w') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print("Confusion Matrix:")
            print(f"True Positives (TP): {self.tp}")
            print(f"True Negatives (TN): {self.tn}")
            print(f"False Positives (FP): {self.fp}")
            print(f"False Negatives (FN): {self.fn}")
            print(f"Accuracy: {self.accuracy()}")
            print(f"Precision: {self.precision()}")
            print(f"Recall buggy: {self.recall_buggy_class()}")#class_1
            print(f"Recall clean: {self.recall_clean_class()}")
            print(f"F1 Score: {self.f1_score()}")
            print(f"G-mean: {self.g_mean()}")
            sys.stdout = original_stdout  # Restore the original stdout
