import pandas as pd
import logging, math, datetime
import csv,json,os,sys,argparse,time,pprint
from CSVData import *
from util import Queue, ConfusionMatrix
from MLT import MoreLikeThisQuery
from MLT import ElasticsearchHandler
# Configure logging
logging.basicConfig(
    filename='example.log',  # Specify the log file name
    level=logging.INFO,      # Set the minimum log level (DEBUG logs everything)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for your module
logger = logging.getLogger(__name__)
project=["tomcat","JGroups","spring-integration",
				"camel","brackets","nova","fabric8",
				"neutron","npm","BroadleafCommerce"]

#the commit is not buggy
NOT_BUG = 0
# the commit is buggy but its true label was not found within W days for test dataset
BUG_NOT_DISCOVERED_W_DAYS = 1
#the commit is buggy and its true label was found within W days for test dataset
BUG_DISCOVERED_W_DAYS = 2
#the true label of a defective commit was assigned.
BUG_FOUND = 3

def read_json(file_path):
	with open(file_path, 'r') as json_file:
		json_document = json.load(json_file)
		return json_document

def get_lines_added(file_path):
    json_document = read_json(file_path)
    return json_document["lines_added"]

def get_files_for_commit(commit_hash,folder_path):
    # Search for files of each commit in the folder
    commit_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_name_split = file_name.split("_", 1)
            if commit_hash == file_name_split[0]:
                commit_files.append(os.path.join(root, file_name))
    return commit_files

def populate_index(es_handler,index_name,folder_path,commit_id,doc_id):
    commit_files = get_files_for_commit(commit_id,folder_path)
    if len(commit_files) > 0:
        for file_path in commit_files:
            document = read_json(file_path)
            doc_id += 1
            response = es_handler.index_json_document(document, index_name, doc_id)
        print("Indexed ",len(commit_files), " docs for ",commit_id)
    return doc_id

def populate_index(es_handler,index_name,folder_path,commit_id,doc_id):
    commit_files = get_files_for_commit(commit_id,folder_path)
    if len(commit_files) > 0:
        bulk_data=[]
        for file_path in commit_files:
            json_doc = read_json(file_path)
            doc_id += 1
            document = {
                "_op_type": "index",
                "_index": index_name,
                "_id": doc_id,
                "_source": json_doc
            }
            bulk_data.append(document)

        response = es_handler.index_bulk(document, index_name, doc_id)


def run_evaluation_with_latency(csv_data_list,type3_list,folder_path,es_handler,index_name):
     TRAIN_DATA_LENGTH = math.ceil(len(csv_data_list)*0.1)
     W = 90
     row_count = 0
     doc_id = 0
     WFL_queue = Queue()
     CLH_queue = Queue()
     result_list = []
     for row in csv_data_list:
         if row_count <= TRAIN_DATA_LENGTH :
             doc_id = populate_index(es_handler,index_name,folder_path,row.commit_id,doc_id)
         elif row_count > TRAIN_DATA_LENGTH: #only index first 10% commits
             predicted = predict(row.commit_id,folder_path,index_name)
             if predicted is not None:
                 res = Result(row.commit_id,row.contains_bug,predicted)
                 result_list.append(res)
                 print(row.commit_id,row.contains_bug,predicted)
             WFL_queue.enqueue(row) #store object of class CSVData
             #find training examples to update model before next timestep
             new_tr_examples = check_WFL_queue(WFL_queue,CLH_queue,row.author_date_unix_timestamp,W,type3_list)

             if len(new_tr_examples) > 0:
                 for example in new_tr_examples:
                    doc_id = populate_index(es_handler,index_name,folder_path,example.commit_id,doc_id)

             new_tr_examples = check_CLH_queue(CLH_queue,row.author_date_unix_timestamp,W,type3_list)
             if len(new_tr_examples) > 0:
                 for example in new_tr_examples:
                     doc_id = populate_index(es_handler,index_name,folder_path,example.commit_id,doc_id)
         row_count +=1
     return result_list

def predict(commit_id,folder_path,index_name):
    mlt_query_executor = MoreLikeThisQuery(index_name) #class object
    commit_files = get_files_for_commit(commit_id,folder_path)
    print("==============================================")
    print("Testing for ",commit_id,"Len commit_files:",len(commit_files))
    if len(commit_files)>0:
        for file_path in commit_files:
            #print("File:",os.path.basename(file_path))
            text = get_lines_added(file_path)
            like_text = text       # Specify the text you want to use for similarity
            field = "lines_added"  # The field in your index to compare against
            min_term_freq = 1
            min_doc_freq = 1
            mlt_query_executor.execute_mlt_query(like_text, field, min_term_freq, min_doc_freq)
        #print_similar_documents(mlt_query_executor.similar_documents)
        clf = Classifier(mlt_query_executor.similar_documents)
        predicted = clf.classify_knn(3)
        return predicted
    return None

def check_WFL_queue(WFL_queue,CLH_queue,current_timestamp,W,type3_list):
    #either bug is reported for commit or it has waited for W days in WFL_queue
    tr_examples = []
    #print("Printing Queue!")
    #WFL_queue.print_queue()
    for row in WFL_queue:
        if row.commit_type==NOT_BUG or row.commit_type==BUG_NOT_DISCOVERED_W_DAYS:
            if calculate_time_elapsed(current_timestamp,row.author_date_unix_timestamp) >= W:
                #train at timestamp+w
                #print(row.commit_id," is clean example at", current_timestamp)
                row.contains_bug = 'False'
                tr_examples.append(row)
                WFL_queue.remove(row) #specific element
                CLH_queue.enqueue(row)

        elif row.commit_type == BUG_DISCOVERED_W_DAYS:#2
            # How do I check when bug was reported?
            # Were any bugs reported by current_timestamp
            # How to create a defect-inducing training_example for commit
            if (defect_linked_at_timestamp(row,current_timestamp,type3_list) is True):
                print(">>>>>>>>>>>>>>>>>>Defect linked to:",row.commit_id,"and current_timestamp is:",current_timestamp)
                row.contains_bug = 'True'
                tr_examples.append(row)
                WFL_queue.remove(row)
    print("Len Tr examples in WFL-Q:",len(tr_examples))
    return tr_examples

def check_CLH_queue(CLH_queue,current_timestamp,W,type3_list):
    tr_examples = []
    for row in CLH_queue:
        if (defect_linked_at_timestamp(row,current_timestamp,type3_list) is True):#1
            #How to swap label of training example
            row.contains_bug = 'True'
            tr_examples.append(row)
            CLH_queue.remove(row)
    print("Len Tr examples in CLH-Q:",len(tr_examples))
    return tr_examples

def defect_linked_at_timestamp(item,current_timestamp,type3_list):
    for row in type3_list:
        if row==item:
            print("timestamp->",row.author_date_unix_timestamp,item.author_date_unix_timestamp)
            if row.author_date_unix_timestamp <=current_timestamp:
                return True
    return False

def print_similar_documents(similar_documents):
    pprint.pprint((similar_documents))
    #for doc in similar_documents:
        #print(f"Document ID: {doc['doc_id']}, Score: {doc['score']} ,commit: {doc['commit']}, filename: {doc['filename']}, commit_filename:{doc['commit_filename']}")
    #print("\n\n")

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

class Result:
    def __init__(self, commit, actual, predicted):
        self.commit = commit
        self.actual = actual
        self.predicted = predicted

def get_w_in_unixtimestamp(timestamp,W):
    #timestamp = 1631086800  # Unix timestamp in seconds for September 8, 2021
    date_obj = datetime.datetime.utcfromtimestamp(timestamp)
    # Calculate a new datetime object 90 days in the future
    new_date_obj = date_obj + datetime.timedelta(days=W)
    # Convert the new datetime object back to a Unix timestamp
    new_timestamp = int(new_date_obj.timestamp())
    return new_timestamp

def calculate_time_elapsed(timestamp1,timestamp2):
    date1 = datetime.datetime.fromtimestamp(timestamp1)
    date2 = datetime.datetime.fromtimestamp(timestamp2)
    time_difference = date1 - date2
    days_difference = time_difference.days
    return days_difference


def main(argv):
    # TO DOs:
    # Dry run
    # Instead of checking at every timestep, may be check on every 10
    # How to get multiple metrics at different timesteps? Just call ConfusionMatrix
    # right after result. line 74-75
    # Test this on camel_commits_test.csv and draw graph
    # Setup JITLine evaluation
    # ensemble of knn and la?
    parser = argparse.ArgumentParser(description="Example script to demonstrate argument parsing.")
    parser.add_argument('arg1', type=str, help='Project name.')
    parser.add_argument('arg2', type=str, help='test arg2')
    #parser.add_argument('--arg2', type=str, default='default_value', help='Description of argument 2')
    args = parser.parse_args()

    project = args.arg1
    index_name = "cabral_"+project.lower()
    path = "/home/hareem/UofA2023/eseval_v2/eseval_timewise/cabral_dataset/"+project+"/data/"
    folder_path = path+project+"_jsonfiles/"
    csv_file = path+project+"_commits.csv"

    es_handler = ElasticsearchHandler()
    response = es_handler.check_health()

    response = es_handler.delete_index(index_name)

    if not es_handler.client.indices.exists(index=index_name):
        response = es_handler.create_index(index_name)
    csv_data_list,type3_list = read_commits_csv(csv_file)
    result_list = run_evaluation_with_latency(csv_data_list,type3_list,folder_path,es_handler,index_name)

    result_file = "/home/hareem/UofA2023/eseval_v2/eseval_timewise/results/metrics_"+project+".txt"
    cm = ConfusionMatrix(result_list)
    cm.display_to_file(result_file)

    response = es_handler.delete_index(index_name)
    print("Index deleted:",response)

    #test_queue()
if __name__ == "__main__":
    main(sys.argv[1:])
