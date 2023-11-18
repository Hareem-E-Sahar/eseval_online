import csv,re,pickle,os,subprocess
import pandas as pd
import time,requests,pprint
import json, git, subprocess
from unidiff.errors import UnidiffParseError
from unidiff import PatchSet
from io import StringIO
import sys,math,warnings
from CSVData import *
warnings.filterwarnings('ignore')

all_tokens_train=[]
all_tokens_test=[]
all_tokens=[]


def make_empty_dict(commit_id,label):
    my_dict2={}
    codelist=[]
    code_dict = {}
    code_dict['added_code'] = []
    code_dict['removed_code'] = []
    codelist.append(code_dict)
    my_dict2 = {
                "commit_id": commit_id,
                "message": "", #dummy text
                "code": codelist,
                "buggy": str(label)
              }
    return my_dict2


def get_patcheSet(repo,commit_id):
    home_directory = os.path.expanduser("~")
    repo_dir_address = os.path.join(home_directory, "UofA2023", repo)
    #owner should not be in path
    #clone_repo(repo,owner,repo_dir_address)
  
    repository = git.Repo(repo_dir_address)
   
    try:
        uni_diff_text = subprocess.check_output(['git', 'diff', '-U0', commit_id+'~1',commit_id,'--'], cwd=repo_dir_address)
        patch_set = PatchSet(StringIO(uni_diff_text.decode("ISO-8859-1")))
	
    except UnidiffParseError as e:
    	return None
 
    except subprocess.CalledProcessError as e:
        return None
    
    return patch_set


def get_data_for_pkl_file(repo,commit_id,label,mode):
    patch_set = get_patcheSet(repo,commit_id)
    codelist=[]     #for entire commit - all patch files
    if patch_set is None or len(patch_set)==0:
        my_dict2 = make_empty_dict(commit_id,label)
        return my_dict2

    for patched_file in patch_set:
        filename = patched_file.path
        addedLines=[]
        addedLinesTokenized=[]
        removedLines=[]
        removedLinesTokenized=[]
        if filename.endswith(".java") or filename.endswith(".py") or filename.endswith(".js"):
            
            for hunk in patched_file:
                for line in hunk:
                    if (line.value.strip() == ''):
                        pass
                    else:
                        if (line.is_added):
                            tokens = tokenize(line.value,mode)
                            addedLinesTokenized.append(' '.join(tokens))

                        elif (line.is_removed):
                            tokens = tokenize(line.value,mode)
                            removedLinesTokenized.append(' '.join(tokens))

        code_dict = {}
        code_dict['added_code'] = addedLinesTokenized
        code_dict['removed_code'] = removedLinesTokenized
        codelist.append(code_dict)

        my_dict = {
                    "commit_id": commit_id,
                    "message": "",
                    "code": codelist,
                    "buggy": str(label)
                  }

        return my_dict #return this if inside for
    return make_empty_dict(commit_id,label)	   #return this if for loop does not execute

def tokenize(line,mode):
    tokens = re.split('([^a-zA-Z0-9])',line)
    nospace_tokens = []
    for tok in tokens:
        tok = tok.strip()
        if (tok):
            nospace_tokens.append(tok)
            if mode=='train':
                all_tokens_train.append(tok)
            if mode=='test':
                all_tokens_test.append(tok)
            all_tokens.append(tok)
    return nospace_tokens


def extract_code_changes_for_pkl(X,repo,mode):

    #X is a list
    print("extract_code_changes_for_pkl:",len(X))
    all_commits=[]
    all_labels=[]
    all_code_changes=[]
    all_messages=[]
    test_commits=[]

    for i in range(len(X)):      # X is list of rows (where each row is represented by a list)
        data_row = X[i]          # Get a single row from csv
        label = data_row[14]     # colname=contains_bug
        commit_id = data_row[17] # colname=commit_id
        #print(commit_id,label)
        my_dict = get_data_for_pkl_file(repo,commit_id,label,mode)

        if my_dict is not None:
            all_commits.append(my_dict['commit_id'])
            all_labels.append(my_dict['buggy'])
            all_messages.append(my_dict['message'])
            all_code_changes.append(my_dict['code'])
        else:
            pass
    print("Length of commits/codes/labels:",len(all_code_changes),len(all_commits),len(all_messages),len(all_labels))
    pkl_data = (all_commits,all_labels,all_messages,all_code_changes)
    return pkl_data
