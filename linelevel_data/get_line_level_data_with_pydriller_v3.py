from pydriller import Repository
from pydriller import Git
import pandas as pd
import csv,os
import multiprocessing



def get_buggy_commits_from_test_data(project,test_data_file):
	df = pd.read_csv(test_data_file)
	buggy_commits_in_test_data = df[df['contains_bug'] == True]['commit_id']
	return buggy_commits_in_test_data.tolist()


def get_correctly_pred_buggy_commits(irjit_file, jitline_file):
	irjit_df = pd.read_csv(irjit_file)

	jitline_df =  pd.read_csv(jitline_file)

	result_jitline_df =  jitline_df[(jitline_df['actual'] == True) & (jitline_df['pred'] == True)]
	result_irjit_df = irjit_df[(irjit_df['actual'] == True) & (irjit_df['predicted'] == True)]

	combined_df = pd.merge(irjit_df, jitline_df , on='commit_id', suffixes=('_irjit', '_jitline'))


	combined_df = combined_df[(combined_df['actual_irjit'] == True) & (combined_df['actual_jitline'] == True) & (combined_df['pred'] == True) & (combined_df['predicted'] == True)]

	# Commits that are true positives in both models
	true_buggy_commits = combined_df[['commit_id']]

	#file_name="/home/hareem/UofA2023/eseval_v2/bugfix_data_using_pydriller/"+project+"_true_buggy_commits.csv"
	#true_buggy_commits.to_csv(file_name, sep=',', encoding='utf-8', index=False)
	return result_irjit_df[['commit_id']],result_jitline_df[['commit_id']],true_buggy_commits



def save_fixing_commits(fixing_commits):
	outfile = project+'_fix_commits.csv'
	with open(outfile, mode='w', newline='') as output_file:
		fieldnames = ['commit_id','date','message']
		writer = csv.DictWriter(output_file, fieldnames)
		writer.writeheader()
		row={}
		for commit in fixing_commits:
			row['commit_id'] = commit.hash
			row['date'] = commit.committer_date
			row['message'] = commit.msg
			#"Files Changed": [mod.filename for mod in commit.modifications],
			#"Code Changes": [mod.diff for mod in commit.modifications]
			writer.writerow(row)

			'''
			print("Files Changed:", commit_info["Files Changed"])
	       	print("Code Changes:", commit_info["Code Changes"])
	       	print("="*100)
			'''


def find_defect_fixing_commits(repo):
	fixing_commits = []

	for commit in repo.traverse_commits():
		if "#" in commit.msg:
			#if commit.hash in buggy_commits:
			fixing_commits.append(commit)
	return fixing_commits


def get_defective_lines(fix_commit, patchfile):
	added_lines=[]
	removed_lines=[]
	for m in fix_commit.modified_files:
		#print(m.filename, m.change_type.name)

		if m.filename == os.path.basename(patchfile):
			#print(m.filename)
			diff = m.diff_parsed
			buggy_lines = []
			clean_lines = []
			if m.change_type.name == 'ADD':
				clean_lines = diff['added']	#[(25, 'import org.junit.Ignore;'), (122, '\t@Ignore @Test')]


			elif m.change_type.name =='DELETE':
				buggy_lines = diff['deleted'] # [(121, '\t@Test')]


			elif m.change_type.name =='MODIFY':
				buggy_lines = diff['deleted']
				clean_lines = diff['added']

			if buggy_lines:
				for line_number, code_line in buggy_lines:
					removed_lines.append(code_line.strip())
			if clean_lines:
				for line_number, code_line in clean_lines:
					added_lines.append(code_line.strip())


	return added_lines, removed_lines


def match_buggy_and_fixing(fixing_commits,buggy_commits_list,writer):
	commits_to_blame = []
	for fix_commit in fixing_commits:
			print(fix_commit.hash)
			#set of commits that changed last the lines modified in the files included in the fix_commit.
			bug_introducing_commits = gitrepo.get_commits_last_modified_lines(fix_commit)
			for key, values in bug_introducing_commits.items():
				if key.endswith('.java') or key.endswith('.py'):
					 #sourcecode modified
					 patchfile = key
					 for buggy_commit in values:
						 if buggy_commit in buggy_commits_list:
							 if(buggy_commit in irjit_pred_buggy_commits):
							 	print("YAY",fix_commit.hash,buggy_commit)
							 added_lines,removed_lines = get_defective_lines(fix_commit,patchfile) #examine diff of fix_commit					 commits_to_blame.append(buggy_commit)
							 if added_lines:
								 add_to_csvfile(fix_commit.hash, buggy_commit, added_lines,   0,'added', writer) #clean
							 if removed_lines:
								 add_to_csvfile(fix_commit.hash, buggy_commit, removed_lines, 1,'deleted', writer)
	return commits_to_blame


def process_commits(fixing_commits,buggy_commits_list,line_level_file):
	header=[ 'fix_commit_hash', 'commit_hash', 'change_type', 'is_buggy_line', 'code_change_remove_common_tokens']
	f = open(line_level_file, "a", newline='')
	writer = csv.writer(f)
	writer.writerow(header)

	commits_to_blame = match_buggy_and_fixing(fixing_commits,buggy_commits_list,writer)
	f.close()
	return commits_to_blame

def add_to_csvfile(fix_commit_id, buggy_commit_id, lines, status, change_type, writer):
	unique_lines=[]
	linenum = 0
	for orignal_line in lines:
		line = orignal_line.strip()
		if line not in unique_lines:
			if(len(line)!=0):
				csvrow = []
				csvrow.append(fix_commit_id)
				csvrow.append(buggy_commit_id)
				csvrow.append(change_type)
				csvrow.append(status)
				csvrow.append(line)
				writer.writerow(csvrow)
		linenum += 1
		unique_lines.append(line)



#identify defect fixing commits for each defect introducing commit in original data.
all_projects=['npm','brackets']
for project in all_projects:
	print(project)
	basedir = "/home/hareem/UofA2023/"
	repo_path = basedir+project
	gitrepo = Git(repo_path)
	repo = Repository(repo_path)

	test_data_file = "/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/online_eval_results/sampled_test_commits/"+project+"_sampled_test_commits.csv"
	#test_data_file = "/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/change_metrics/"+project+"_metrics.csv"
	buggy_commits = get_buggy_commits_from_test_data(project,test_data_file)
	'''
	jitline_file="/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/online_eval_results/predictions/"+project+"_RF_DE_SMOTE_min_df_3_prediction_result.csv"
	irjit_file = "/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results/results_lines_added_camel/resultlist_"+project+"_K=3.csv"
	irjit_pred_buggy_commits,jitline_pred_buggy_commits,combined_pred_buggy_commits = get_correctly_pred_buggy_commits(irjit_file,jitline_file)
	print(len(buggy_commits),len(irjit_pred_buggy_commits),len(jitline_pred_buggy_commits),len(combined_pred_buggy_commits))
	'''
	fix_commits = find_defect_fixing_commits(repo)
	save_fixing_commits(fix_commits)
	line_level_file = "/home/hareem/UofA2023/eseval_v2/eseval_online/linelevel_data/"+project+"_complete_buggy_line_level.csv"
	commits_to_blame = process_commits(fix_commits,buggy_commits,line_level_file)
	commits_to_blame_df = DataFrame(commits_to_blame)
	unique_commits_to_blame_df = commit_to_blame_df['commit_hash'].unique()
	unique_commits_to_blame_df.to_csv("./"+project+"_buggy_commits_to_blame.csv")
	print("All done!")


#https://stackoverflow.com/questions/70231772/how-to-get-last-touched-lines-in-a-file-of-a-commit-pydriller
#https://pydriller.readthedocs.io/en/latest/reference.html
#https://pydriller.readthedocs.io/en/latest/git.htmltomcat
