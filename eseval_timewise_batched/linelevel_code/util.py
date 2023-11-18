import sys,os,json
from ConfusionMatrix import *

def read_json(file_path):
	with open(file_path, 'r',encoding='utf-8') as json_file:
		json_document = json.load(json_file)
		return json_document

def get_lines_added(file_path):
    json_document = read_json(file_path)
    return json_document["lines_added"]

def get_lines_deleted(file_path):
    json_document = read_json(file_path)
    return json_document["lines_deleted"]

def get_files_for_commit(commit_hash,folder_path):
    # Search for files of each commit in the folder
    commit_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_name_split = file_name.split("_", 1)
            if commit_hash == file_name_split[0]:
                commit_files.append(os.path.join(root, file_name))
    return commit_files

def save_result(project,K,cm,execution_time,results_dir):
	predictions_file = os.path.join(results_dir, f"resultlist_{project}_K={K}.csv")
	cm.save_result_list(predictions_file)
	metrics_file = os.path.join(results_dir, f"metrics.csv")
	header = ["project", "K", "TP", "TN", "FP", "FN", "Prec", "Recall1", "Recall0", "F1", "Gmean", "Acc","time"]
	metrics = [project, K, cm.tp, cm.tn, cm.fp, cm.fn, cm.precision(), cm.recall_buggy_class(), cm.recall_clean_class(), cm.f1_score(), cm.g_mean(), cm.accuracy(),execution_time]
	file_exists = os.path.isfile(metrics_file)
	with open(metrics_file, 'a') as file:
		writer = csv.writer(file)
		if not file_exists:
			writer.writerow(header)
		writer.writerow(metrics)

'''

import cProfile
import pstats
from functools import wraps
def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval
        return wrapper
    return inner
'''