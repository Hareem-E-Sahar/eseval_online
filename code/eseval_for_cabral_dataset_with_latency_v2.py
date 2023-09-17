import pandas as pd
import logging, math, datetime
import csv,json,os,sys,argparse,time,pprint
from CSVData import *
from util import *
from MLT import *
from Queue import Queue
from Classifier import Classifier
from elasticsearch.helpers import bulk
import cProfile, pstats, io
from pstats import SortKey

commit_files_dict = {}
'''
# Configure logging
logging.basicConfig(
    filename='example.log',  # Specify the log file name
    level=logging.INFO,      # Set the minimum log level (DEBUG logs everything)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Create a logger for your module
logger = logging.getLogger(__name__)
'''
projects=["tomcat","JGroups","spring-integration",
				"camel","nova","fabric8",
				"neutron","BroadleafCommerce"]
#the commit is not buggy
NOT_BUG = 0
# the commit is buggy but its true label was not found within W days for test dataset
BUG_NOT_DISCOVERED_W_DAYS = 1
#the commit is buggy and its true label was found within W days for test dataset
BUG_DISCOVERED_W_DAYS = 2
#the true label of a defective commit was assigned.
BUG_FOUND = 3

def populate_index_bulk(es_handler,index_name,folder_path,commit_id,doc_id):
    if commit_id in commit_files_dict:
        commit_files = commit_files_dict[commit_id]
    else:
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
        response = es_handler.index_bulk(bulk_data)
    return doc_id

#@profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
def run_evaluation_with_latency(csv_data_list,type3_dict,folder_path,es_handler,index_name,K):
     TRAIN_DATA_LENGTH = math.ceil(len(csv_data_list)*0.1)
     W = 90
     row_count = 0
     doc_id = 0
     WFL_queue = Queue()
     CLH_queue = Queue()
     result_list = []
     for row in csv_data_list:
         if row_count <= TRAIN_DATA_LENGTH :#only index first 10% commits
             doc_id = populate_index_bulk(es_handler,index_name,folder_path,row.commit_id,doc_id)
         elif row_count > TRAIN_DATA_LENGTH:
             predicted = predict(K,row.commit_id,folder_path,index_name)
             if predicted is not None:
                 res = Result(row.commit_id,row.contains_bug,predicted)
                 result_list.append(res)
                 #print(row.commit_id,row.contains_bug,predicted)
             WFL_queue.enqueue(row) #store object of class CSVData
             #find training examples to update model before next timestep
             new_tr_examples = check_WFL_queue(WFL_queue,CLH_queue,row.author_date_unix_timestamp,W,type3_dict)
             if len(new_tr_examples) > 0:
                 for example in new_tr_examples:
                    doc_id = populate_index_bulk(es_handler,index_name,folder_path,example.commit_id,doc_id)

             new_tr_examples = check_CLH_queue(CLH_queue,row.author_date_unix_timestamp,W,type3_dict)
             if len(new_tr_examples) > 0:
                 for example in new_tr_examples:
                     doc_id = populate_index_bulk(es_handler,index_name,folder_path,example.commit_id,doc_id)
         row_count +=1
     return result_list

#parallelize
def predict(K,commit_id,folder_path,index_name):
    mlt_query_executor = MoreLikeThisQuery(index_name) #class object
    commit_files = get_files_for_commit(commit_id,folder_path)
    commit_files_dict[commit_id] = commit_files
    print("Test commit:",commit_id)
    if len(commit_files)>0:
        for file_path in commit_files:
            #print("File:",os.path.basename(file_path))
            like_text = get_lines_added(file_path)       # Specify the text you want to use for similarity
            field = "lines_added"  # The field in your index to compare against
            mlt_query_executor.execute_mlt_query(like_text, field) #min_term_freq, min_doc_freq
        #print_similar_documents(mlt_query_executor.similar_documents)
        clf = Classifier(mlt_query_executor.similar_documents)
        predicted = clf.classify_knn(K)
        return predicted
    return None

#@profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
def check_WFL_queue(WFL_queue,CLH_queue,current_timestamp,W,type3_dict):
    #either bug is reported for commit or it has waited for W days in WFL_queue
    tr_examples = []
    #print("Printing Queue!")
    #WFL_queue.print_queue()
    for row in WFL_queue:
        if row.commit_type==NOT_BUG or row.commit_type==BUG_NOT_DISCOVERED_W_DAYS:
            if calculate_time_elapsed(current_timestamp,row.author_date_unix_timestamp) >= W:
                #print(row.commit_id," is clean example at", current_timestamp)
                row.contains_bug = 'False'
                tr_examples.append(row)
                WFL_queue.remove(row) #specific element
                CLH_queue.enqueue(row)

        elif row.commit_type == BUG_DISCOVERED_W_DAYS:#2
            # How do I check when bug was reported?
            # Were any bugs reported by current_timestamp
            # How to create a defect-inducing training_example for commit
            if (defect_linked_at_timestamp(row,current_timestamp,type3_dict) is True):
                #print(">>>>>>>>>>>>>>>>>>Defect linked to:",row.commit_id,"and current_timestamp is:",current_timestamp)
                row.contains_bug = 'True'
                tr_examples.append(row)
                WFL_queue.remove(row)
    return tr_examples

#@profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
def check_CLH_queue(CLH_queue,current_timestamp,W,type3_dict):
    tr_examples = []
    for row in CLH_queue:
        if (defect_linked_at_timestamp(row,current_timestamp,type3_dict) is True):#1
            row.contains_bug = 'True'
            tr_examples.append(row)
            CLH_queue.remove(row)
    return tr_examples

def defect_linked_at_timestamp(item,current_timestamp,type3_dict):
    if item.commit_id in type3_dict:
        defect_timestamp  = type3_dict[item.commit_id]
        #print("timestamp->",defect_timestamp,item.author_date_unix_timestamp)
        if defect_timestamp <= current_timestamp:
            return True
    return False

def print_similar_documents(similar_documents):
    pprint.pprint((similar_documents))
    #for doc in similar_documents:
        #print(f"Document ID: {doc['doc_id']}, Score: {doc['score']} ,commit: {doc['commit']}, filename: {doc['filename']}, commit_filename:{doc['commit_filename']}")
    #print("\n\n")

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

#@profile(sort_by='cumulative', lines_to_print=20, strip_dirs=True)
def main(argv):
    # TO DOs:
    # Dry run
    # Imp:Instead of checking at every timestep, may be check on every 10
    # How to get multiple metrics at different timesteps? Just call ConfusionMatrix
    # after updating resultlist.
    # Test this on camel_commits_test.csv and draw graph
    # Setup JITLine evaluation
    # ensemble of knn and la?
    parser = argparse.ArgumentParser(description="Example script to demonstrate argument parsing.")
    parser.add_argument('-project', type=str,  default='', help='Project name.')
    parser.add_argument('-K', type=int, default=3, help='value of K for KNN.')
    args = parser.parse_args()

    current_dir = os.getcwd() #code
    parent_dir = os.path.dirname(current_dir) #eseval_online
    print(parent_dir)
    results_dir = os.path.join(parent_dir, "results") #results dir
    data_dir = os.path.join(parent_dir, "cabral_dataset",args.project,"data/")
    print(data_dir)


    index_name = "cabral_"+args.project.lower()
    jsonfolder_path = os.path.join(data_dir , f"{args.project}_jsonfiles")
    csv_file    = os.path.join(data_dir , f"{args.project}_commits.csv")
    print(jsonfolder_path)
    print(csv_file)
    es_handler = ElasticsearchHandler()
    #response = es_handler.check_health()
    response = es_handler.delete_index(index_name)
    if not es_handler.client.indices.exists(index=index_name):
        response = es_handler.create_index(index_name)
        print(response)

    csv_data_list,type3_list = read_commits_csv(csv_file)
    type3_dict = find_matches(csv_data_list,type3_list)

    start_time = time.time()
    result_list = run_evaluation_with_latency(csv_data_list,type3_dict,jsonfolder_path,es_handler,index_name,args.K)
    end_time = time.time()

    execution_time = end_time - start_time
    cm = ConfusionMatrix(result_list)
    cm.compute_metrics()
    save_result(args.project,args.K,cm,execution_time,results_dir)

    response = es_handler.delete_index(index_name)
    print("Index deleted:",response)

if __name__ == "__main__":
    main(sys.argv[1:])
