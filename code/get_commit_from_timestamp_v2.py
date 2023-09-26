from datetime import datetime
from dateutil.parser import parse
import csv, subprocess
import pytz, argparse
from github import Github
from dateutil import parser
from datetime import datetime
'''
This code is NOT memory intensive because it writes one row at a time to the file.
python3 timestamp_to_commit.py -r ../camel -p "../suppmaterial-19-dayi-risk_data_merging_jit/data/camel2.csv" -n "camel"
'''

data=[]
def read_and_analyze_data(fname):
	with open(fname, 'r') as csvfile:
		csvreader = csv.DictReader(csvfile)
		for row in csvreader:
			print(row['fixes'])

def commits_by_type(fname):
	import pandas as pd
	import numpy as np
	df=pd.read_csv(fname)
	print("Total rows:",len(df))
	unique_values=df["commit_type"].unique()
	specific_column=df["commit_type"].value_counts()
	print(specific_column)
	#specific_column=df["contains_bug"].value_counts()


def get_commit_hashes(fname,wfname):
	commit_dicts = get_gitlog()
	print("len of gitlog:",len(commit_dicts))
	with open(fname, mode='r') as input_file, open(wfname, mode='w', newline='') as output_file:
		reader = csv.DictReader(input_file)
		fieldnames = reader.fieldnames + ['commit_id']
		writer = csv.DictWriter(output_file, fieldnames)
		writer.writeheader()
		i=0
		for row in reader:
			i=i+1
			if(i%10==0):
				print("Progress..."+str(i))
			actual_timestamp = row['author_date_unix_timestamp']
			if actual_timestamp is not None:
				commit_sha = timestamp_to_commit_using_gitlog(int(actual_timestamp))
				if len(commit_sha) == 0 :
					commit_sha = fetch_commit_hash_from_gitlog(str(actual_timestamp),commit_dicts)
				row['commit_id'] = commit_sha
				writer.writerow(row)

'''
def timestamp_to_commit_using_API(actual_timestamp):
	access_token = "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"
	g = Github(access_token)
	repo = g.get_repo("apache/camel")

	start_datetime = datetime.fromtimestamp(actual_timestamp-60,tz=pytz.UTC)
	end_datetime = datetime.fromtimestamp(actual_timestamp+60,tz=pytz.UTC)
	print("start time",start_datetime)
	print("end time",end_datetime)

	#git log --after="2017-11-29 10:40:05" --before="2017-11-29 10:40:05"
	commits = repo.get_commits(since=start_datetime, until=end_datetime)
	for commit in commits:
		return commit.sha #in case of multiple, first is returned
'''

def timestamp_to_commit_using_gitlog(actual_timestamp):
	#This way you don't encounter API rate limit error.
	#dt_obj = datetime.fromtimestamp(actual_timestamp,tz=pytz.UTC)
	start_datetime = datetime.fromtimestamp(actual_timestamp-60,tz=pytz.UTC)
	end_datetime = datetime.fromtimestamp(actual_timestamp+60,tz=pytz.UTC)
	command = f"git log --pretty=format:'%H' --after='{start_datetime}' --before='{end_datetime}'"
	output = subprocess.check_output(command, shell=True).decode()
	# Split the output into individual commit hashes
	commit_hashes = output.strip().split('\n')
	if commit_hashes:
	    return commit_hashes[0]	#in case of multiple, first is returned
	else:
		return None

def find_unique():
	import pandas as pd
	import numpy as np
	df=pd.read_csv(fname)
	specific_column=df["classification"].unique()
	#specific_column=df["contains_bug"].value_counts()

import json, git, subprocess, pathlib, getopt
from unidiff import PatchSet
from io import StringIO
import pandas as pd
import time, csv, os, sys
from csv import DictReader

def createFileIfNotExist(outfilename):
    if not os.path.exists(os.path.dirname(outfilename)):
        try:
            os.makedirs(os.path.dirname(outfilename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc.errno)

def get_patchset(commit_id,project,repoFolder):
    repository = git.Repo(repoFolder)
    commit = repository.commit(commit_id)
    try:
        uni_diff_text = subprocess.check_output(['git', 'diff', '-U0', commit_id+'~1',commit_id ], cwd=repoFolder)
        patch_set = PatchSet(StringIO(uni_diff_text.decode("ISO-8859-1")))
        return patch_set
    except:
        return None

def get_output_file_name(dir, commit_id, filename):
	if filename.endswith('.java'):
		trimlen = 5
	elif filename.endswith('.py'):
		trimlen = 3
	elif filename.endswith('.c'):
		trimlen = 2
	elif filename.endswith('.cpp'):
		trimlen = 4
	fname = dir + commit_id + '_' + os.path.basename(filename)[:-trimlen] + '.json'
	return fname

def curl_commits(commit_id,project,buggy_status,dir,repositoryFolder):
	patch_set = get_patchset(commit_id,project,repositoryFolder)
	if patch_set is None:
		pass
	else:
		for patched_file in patch_set:
			filename = patched_file.path
			if filename.endswith('.py') or filename.endswith('.java') :
				addedLines=[]
				removedLines=[]
				for hunk in patched_file:
					for line in hunk:
						if line.is_added and line.value.strip() != '':
							addedLines.append(line.value)
						if line.is_removed and line.value.strip() != '':
							removedLines.append(line.value)
				my_dict= {
					"commit_id": commit_id,
					"filename": filename,
					"lines_added": "".join(addedLines),
					"lines_deleted": "".join(removedLines),
					"buggy":buggy_status,
				}
				fname = get_output_file_name(dir,commit_id,filename)
				#print("=== Writing to ",fname)
				with open(fname,"w") as outfile:
					json.dump(my_dict, outfile)

def get_commit_sourcecode(fname,dir,project,repoFolder):
	processed_commits = set()
	with open(fname,'r') as csv_file:
		csv_reader = csv.DictReader(csv_file, delimiter=',')
		#header = next(csv_reader,None)
		total_commits_processed = 0
		for row in csv_reader:
			commit_id = row["commit_id"]
			if commit_id not in processed_commits and len(commit_id)>0:
				print(commit_id)
				label = row["contains_bug"]
				#ctimestamp = int(row["author_date_unix_timestamp"])
				#cdate = datetime.fromtimestamp(ctimestamp,tz=pytz.UTC)
				total_commits_processed += 1
				processed_commits.add(commit_id)
				curl_commits(commit_id,project,label,dir,repoFolder)
	print("Done fetching testing code diffs, total: ",total_commits_processed)

def date_to_timestamp(iso_date):
    # Parse the ISO date string into a datetime object
    date_object = parser.parse(iso_date)
    # Convert the datetime object to a Unix timestamp (in seconds)
    unix_timestamp = int(date_object.timestamp())
    return int(unix_timestamp)


def fetch_commit_hash_from_gitlog(actual_timestamp,commit_dict_list):
	for commit_dict in commit_dict_list:
		if commit_dict['author_timestamp']==actual_timestamp:
			return (commit_dict['commit_hash'])

def get_gitlog():
	git_log_output = subprocess.check_output(['git', 'log', '--pretty=format:%H, %ct, %ci, %ad', '--date=iso'], text=True)
	# Split the git log output into lines
	lines = git_log_output.strip().split('\n')

	commit_dicts = []
	# Iterate through each line of the git log output
	for line in lines:
	    # Split the line into commit hash and date strings
	    commit_hash, date_strs = line.split(',', 1)
	    date_strs = date_strs.strip().split(',')

	    commit_dict = {
	        'commit_hash': str(commit_hash),
	        'timestamp': str(date_strs[0]),
	        'commit_date': str(date_strs[1]),
	        'author_date': str(date_strs[2]),
	        'commit_timestamp': str(date_to_timestamp(date_strs[1])),
	        'author_timestamp': str(date_to_timestamp(date_strs[2]))
	    }
	    # Append the commit dictionary to the list
	    commit_dicts.append(commit_dict)
	return commit_dicts

def test1():
	print(timestamp_to_commit_using_gitlog(1404477609))
	commit_dict = get_gitlog()
	print(fetch_commit_hash_from_gitlog(str(1404477609),commit_dict))

def main(argv):
	repositoryFolder = ''

	try:
		opts, args = getopt.getopt(argv,"hr:p:n:")
	except getopt.GetoptError as e:
		print("script.py -r <repositoryFolder> -p <pathToCsv> -n <nameOfTheProject>")
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print("script.py -r <repositoryFolder> -p <pathToCsv> -n <nameOfTheProject>")
			sys.exit()
		if opt == '-p':
			pathToCsv = arg
		elif opt == '-r':
			repoFolder = arg
		elif opt == '-n':
			project = arg

	parentdir ="/home/hareem/UofA2023/eseval_v2/eseval_timewise/cabral_dataset/"+project+"/data/"
	createFileIfNotExist(parentdir)
	subdir = project +"_jsonfiles/"
	jsondir = os.path.join(parentdir, subdir)
	createFileIfNotExist(jsondir)
	write_fname = parentdir+project+'_commits.csv'
	#only call this function to get commit_ids
	get_commit_hashes(pathToCsv,write_fname)

	#call this function to make json files
	#replace first argument (write_fname) with pathToCsv when called without 'get_commit_hashes'
	get_commit_sourcecode(write_fname,jsondir,project,repoFolder)

if __name__ == "__main__":
    main(sys.argv[1:])
