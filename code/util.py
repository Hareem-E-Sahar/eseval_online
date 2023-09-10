import sys
class ConfusionMatrix:
    def __init__(self, result):
        self.result = result
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.compute_metrics()

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

    def display_to_file(self, filename):
        with open(filename, 'w') as file:
            original_stdout = sys.stdout
            sys.stdout = file

            print("Confusion Matrix:")
            print(f"True Positives (TP): {self.tp}")
            print(f"True Negatives (TN): {self.tn}")
            print(f"False Positives (FP): {self.fp}")
            print(f"False Negatives (FN): {self.fn}")
            print(f"Accuracy: {self.accuracy()}")
            print(f"Precision: {self.precision()}")
            print(f"Recall buggy: {self.recall_buggy_class()}")
            print(f"Recall clean: {self.recall_clean_class()}")
            print(f"F1 Score: {self.f1_score()}")
            print(f"G-mean: {self.g_mean()}")
            sys.stdout = original_stdout  # Restore the original stdout

class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            print("*** Dequeue ",self.items[0].commit_id)
            return self.items.pop(0) #element at the front of queue
        else:
            raise IndexError("Queue is empty")

    def get_first(self):
        if not self.is_empty():
            print("$$$ First ",self.items[0].commit_id)

    def size(self):
        return len(self.items)

    def contains(self, item):
        for index, element in enumerate(self.items):
            if element == item:
                return index, element  # Return both the index and the matched element
        return None, None  # Return None if the item is not found

    def remove(self, item):
        index, _ = self.contains(item)
        if index is not None:
            del self.items[index]
            
    def __iter__(self):
        return iter(self.items)

    def print_queue(self):
        print("Elements in Queue!")
        for index, element in enumerate(self.items):
            print(index,element.commit_id)
        print("Queue finished!")

def test_queue():
    my_queue = Queue()
    print(my_queue.is_empty())  # True
    # Enqueue some elements
    my_queue.enqueue(1)
    my_queue.enqueue(2)
    my_queue.enqueue(3)
    my_queue.check_queued_elements()
    print("Size",my_queue.size())  # 3
    print("Element",my_queue.dequeue())  # 1
    print("Element",my_queue.dequeue())  # 2
    print("Size",my_queue.size())   # 1
    # Dequeue the last element
    print("Element",my_queue.dequeue())  # 3
    print(my_queue.is_empty())  # True

def read_commits_csv_with_pandas(csv_file_path):
     na_values = ['NA', 'N/A', 'NaN', 'None', '']
     df = pd.read_csv(csv_file_path,na_values=na_values)
     # you can filter rows based on a condition:
     # buggy_commits = df[df['contains_bug'] == True]
     # label = df[df['commit_id']==id]
     df = df.dropna()
     return df
