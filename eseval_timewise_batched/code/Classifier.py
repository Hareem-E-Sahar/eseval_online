class Classifier:
    def __init__(self, similar_documents):
        self.similar_documents = similar_documents

    def select_topk_docs(self,K):
        topK = self.similar_documents[:K]
        return topK

    def classify_knn(self,K): #assign a final label
        topK = self.select_topk_docs(K)
        label = self.get_majority_label_of_topk(topK)
        return label

    def classify_threshold(self,T):
        print("I need a threshold!")

    def get_majority_label_of_topk(self, topK):
        buggy = 0
        clean = 0
        for doc in topK:
            if doc['label']=='True':
                buggy += 1
            else:
                clean += 1
        return 'True' if buggy > clean else 'False'

