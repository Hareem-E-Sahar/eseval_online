import pandas as pd
import numpy as np
import logging, math, datetime
import csv,json,os,sys,argparse,time,pprint
from CSVData import *
from util import *
from MLT import *
from Queue import Queue
from Classifier import Classifier
from elasticsearch.helpers import bulk
import cProfile, pstats, io
from lineEval import *

commit_files_dict = {}
project = "tomcat"
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


def get_lines_added_to_commit(commit):
	commit_files = commit_files_dict[commit]

	lines_added = []
	for file_path in commit_files:
		lines_added_str = get_lines_added(file_path)
		lines_added = lines_added_str.splitlines() #str to list of lines
	return lines_added


def run_evaluation_with_latency(csv_data_list, type3_dict, folder_path, es_handler, index_name, project, K, sampled_test_data, qtype):
	TRAIN_DATA_LENGTH = math.ceil(len(csv_data_list)*0.1)
	W = 90
	doc_id = 0
	WFL_queue = Queue()
	CLH_queue = Queue()
	result_list = []
	new_tr_examples = []
	commit_vs_buggy_tokens_dict = {}
	commit_vs_buggy_lines = {}
	all_buggy_line_result_df = []
      
	       
	for row in csv_data_list:
		if row.commit_id not in sampled_test_data['commit_id'].values:
			doc_id = populate_index_bulk(es_handler,index_name,folder_path,row.commit_id,doc_id)
			

		else:
			predicted, commit_buggy_tokens  = predict(K,row.commit_id,row.contains_bug,folder_path,index_name,qtype)
			#print(row.commit_id,row.contains_bug,predicted)
			
			if row.contains_bug=='True' and predicted=='True':
				line_score_df = rank_ground_truth_lines(project, row.commit_id, commit_buggy_tokens)
				if line_score_df is not None:
					all_buggy_line_result_df.append(line_score_df)                            
			
			res = Result(row.commit_id,row.contains_bug,predicted) #previously, we checked if predicted is not None
			result_list.append(res)
			
			duplicate_count = np.count_nonzero(sampled_test_data['commit_id'].values == row.commit_id)
			result_list.extend([res] * (duplicate_count - 1)) #add to list count-1 times

			WFL_queue.enqueue(row) #store object of class CSVData
			new_tr_examples = check_WFL_queue(WFL_queue,CLH_queue,row.author_date_unix_timestamp,W,type3_dict)
			new_tr_examples.extend ( check_CLH_queue(CLH_queue,row.author_date_unix_timestamp,W,type3_dict) )

			if len(new_tr_examples) > 0:
				for row in new_tr_examples:
					doc_id = populate_index_bulk(es_handler,index_name,folder_path,row.commit_id,doc_id)
	print("all_buggy_line_result_df:",len(all_buggy_line_result_df))
	return result_list,all_buggy_line_result_df


def predict(K,commit_id,contains_bug,folder_path,index_name,qtype):
	mlt_query_executor = MoreLikeThisQuery(index_name) #class object
	commit_files = get_files_for_commit(commit_id,folder_path)
	commit_files_dict[commit_id] = commit_files
	#print("Test commit:",commit_id)
	predicted = 'False'
	commit_buggy_tokens = []
	if len(commit_files)>0:
		for file_path in commit_files:
			like_text = get_lines_added(file_path)       # text to use for similarity
			field = "lines_added"  		     # field in index to compare against
			like_text2 = get_lines_deleted(file_path)
			field2 = "lines_deleted"

			if qtype == "boolean":
				mlt_query_executor.execute_mlt_query_bool(like_text, field, like_text2, field2)
			else:
				mlt_query_executor.execute_mlt_query(like_text, field) #min_term_freq, min_doc_freq

		commit_buggy_tokens = mlt_query_executor.exp_obj.get_explanation_tokens()
		#print_similar_documents(mlt_query_executor.similar_documents)

		clf = Classifier(mlt_query_executor.similar_documents)
		predicted = clf.classify_knn(K)

	return predicted,commit_buggy_tokens  #'True','False' but not None anymore


def check_WFL_queue(WFL_queue,CLH_queue,current_timestamp,W,type3_dict):
    tr_examples = []
    for row in WFL_queue:
        if row.commit_type==NOT_BUG or row.commit_type==BUG_NOT_DISCOVERED_W_DAYS:
            if calculate_time_elapsed(current_timestamp,row.author_date_unix_timestamp) >= W:
                #print(row.commit_id," is clean example at", current_timestamp)
                row.contains_bug = 'False'
                tr_examples.append(row)
                WFL_queue.remove(row) #specific element
                CLH_queue.enqueue(row)

        elif row.commit_type == BUG_DISCOVERED_W_DAYS:#2
            if (defect_linked_at_timestamp(row,current_timestamp,type3_dict) is True):
                #print(">>>>>>>>>>>>>>>>>>Defect linked to:",row.commit_id,"and current_timestamp is:",current_timestamp)
                row.contains_bug = 'True'
                tr_examples.append(row)
                WFL_queue.remove(row)
    return tr_examples


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


def calculate_time_elapsed(timestamp1,timestamp2):
    date1 = datetime.datetime.fromtimestamp(timestamp1)
    date2 = datetime.datetime.fromtimestamp(timestamp2)
    time_difference = date1 - date2
    days_difference = time_difference.days
    return days_difference


def main(argv):
	parser = argparse.ArgumentParser(description="Pass arguments -p project -K integervalue.")
	parser.add_argument('-project', type=str,  default='', help='Project name.')
	parser.add_argument('-K', type=int, default=3, help='value of K for KNN.')
	parser.add_argument('-settings', type=str, default='', help='index settings needed.')
	parser.add_argument('-querytype', type=str, default='', help='boolean or notboolean')

	args = parser.parse_args()

	current_dir = os.getcwd() 				  #code
	parent_dir = os.path.dirname(current_dir) #eseval_online
	print(parent_dir)
	results_dir = os.path.join(parent_dir, "results") #results dir
	data_dir = os.path.join(parent_dir, "cabral_dataset",args.project,"data/")
	linelevel_results_dir = os.path.join(parent_dir, "results_linelevel/") #results dir
      
	print(data_dir)

	index_name = "cabral_"+args.project.lower()
	jsonfolder_path = os.path.join(data_dir , f"{args.project}_jsonfiles")
	csv_file    = os.path.join(data_dir , f"{args.project}_commits.csv")
	index_settings = os.path.join(current_dir, f"{args.settings}")
	print(jsonfolder_path)
	print(csv_file)

	es_handler = ElasticsearchHandler()
	#response = es_handler.check_health()
	#response = es_handler.delete_index(index_name)
	if not es_handler.client.indices.exists(index=index_name):
		response = es_handler.create_index(index_name,index_settings)
		print(response)


	csv_data_list,type3_list = read_commits_csv(csv_file)
	type3_dict = find_matches(csv_data_list,type3_list)

	jitline_results_folder = "/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/online_eval_results/sampled_test_commits/"


	test_commits_file = jitline_results_folder + "./"+args.project+"_sampled_test_commits.csv"
	sampled_test_data = pd.read_csv(test_commits_file,header=0)
	
    
	start_time = time.time()
	result_list, all_buggy_line_result_df = run_evaluation_with_latency(csv_data_list, type3_dict, jsonfolder_path, es_handler, index_name, args.project , args.K, sampled_test_data, args.querytype)
	end_time = time.time()
	execution_time = end_time - start_time
	cm = ConfusionMatrix(result_list)
	cm.compute_metrics()
	#save_result(args.project,args.K,cm,execution_time,results_dir)
	save_linelevel_result(args.project, args.K, cm, linelevel_results_dir, all_buggy_line_result_df)
	eval_line_level_at_commit(args.project, linelevel_results_dir)
	
	response = es_handler.delete_index(index_name)
	print("Index deleted:",response)

    

if __name__ == "__main__":
    main(sys.argv[1:])
