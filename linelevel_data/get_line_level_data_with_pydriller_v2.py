from pydriller import Repository
from pydriller import Git
import pandas as pd
import csv,os
import multiprocessing


'''
def get_buggy_commits_from_openstack_data():
	filename =  '/home/hareem/Desktop/openstack_RF_DE_SMOTE_min_df_3_prediction_result.csv'
	df = pd.read_csv(filename)
	buggy_commits_in_test_data = df[(df['actual'] == 1) & (df['pred'] == 1)]['test_commit']
	return buggy_commits_in_test_data.tolist()


def unique_buggy_commits_in_openstack_data():
	openstack_line='/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/openstack_and_qt_data/openstack_complete_buggy_line_level.csv'
	
	buggy = get_buggy_commits_from_openstack_data()
	df = pd.read_csv(openstack_line)
	unique_linelevel=df['commit_hash'].unique()
	print(len(unique_linelevel))
	common = set(buggy).intersection(unique_linelevel)
	print(len(buggy),len(unique_linelevel),len(common))


def unique_buggy_commits_in_linelevel_data():
	buggy = get_buggy_commits_from_test_data("tomcat")
	filename = "/home/hareem/UofA2023/eseval_v2/eseval_online/linelevel_data/tomcat_complete_buggy_line_level.csv"
	df = pd.read_csv(filename)
	unique_linelevel=df['commit_hash'].unique()
	common = set(buggy).intersection(unique_linelevel)
	print(len(buggy),len(unique_linelevel),len(common))
'''

def get_buggy_commits_from_test_data(project):
	filename = "/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/online_eval_results/sampled_test_commits/"+project+"_sampled_test_commits.csv"
	df = pd.read_csv(filename)
	buggy_commits_in_test_data = df[df['contains_bug'] == True]['commit_id']
	return buggy_commits_in_test_data.tolist()


def get_correctly_pred_buggy_commits():
	irjit_df = pd.read_csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results/results_lines_added_camel/resultlist_"+project+"_K=3.csv")

	jitline_df =  pd.read_csv("/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/online_eval_results/predictions/"+project+"_RF_DE_SMOTE_min_df_3_prediction_result.csv")

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
							 added_lines,removed_lines = get_defective_lines(fix_commit,patchfile) #examine diff of fix_commit
							 if added_lines:
								 add_to_csvfile(fix_commit.hash, buggy_commit, added_lines,   0,'added', writer) #clean
							 if removed_lines:
								 add_to_csvfile(fix_commit.hash, buggy_commit, removed_lines, 1,'deleted', writer)    #buggy


def process_commits(fixing_commits,buggy_commits_list): #
	line_level_file = "/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/linelevel_data/"+project+"_complete_buggy_line_level_NEW.csv"
	header=[ 'fix_commit_hash', 'commit_hash', 'change_type', 'is_buggy_line', 'code_change_remove_common_tokens']
	f = open(line_level_file, "a", newline='')
	writer = csv.writer(f)
	writer.writerow(header)
	
	match_buggy_and_fixing(fixing_commits,buggy_commits_list,writer)
	
	print("All done!")
	f.close()

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
all_projects=['tomcat'] 
unique_buggy_commits_in_linelevel_data()
'''
for project in all_projects:
	print(project)
	basedir = "/home/hareem/UofA2023/"
	repo_path = basedir+project
	gitrepo = Git(repo_path)
	repo = Repository(repo_path)

	buggy_commits = get_buggy_commits_from_test_data(project)
	irjit_pred_buggy_commits,jitline_pred_buggy_commits,combined_pred_buggy_commits = get_correctly_pred_buggy_commits()
	print(len(buggy_commits),len(irjit_pred_buggy_commits),len(jitline_pred_buggy_commits),len(combined_pred_buggy_commits))
	#tomcat = 353, 317, 187, 176 out of 1200 sampled commits

	
	fix_commits = find_defect_fixing_commits(repo)
	#save_fixing_commits(fix_commits)
	process_commits(fix_commits,buggy_commits)
	#process_commits(fix_commits, irjit_pred_buggy_commits)

'''

#get_defective_lines(fix_commits[0],'spring-integration-ip/src/test/java/org/springframework/integration/ip/tcp/SimpleTcpNetOutboundGatewayTests.java')
'''
The output from get_commits_last_modified_lines
{'pom.xml': {'3be8f7b46d22258029c7b39856ac7f5ee8d7b8d0', 'fa7d80aaaeda11d5d8b1b947941ba50920e48439'}, 'spring-integration-core/pom.xml': {'e4dc5e41e788c03fb971c2c491018e013ea0ae04', '43310c9cabd078310b5d8c5881467f0628c5bdb8'}, 'spring-integration-ip/src/test/java/org/springframework/integration/ip/tcp/SimpleTcpNetOutboundGatewayTests.java': {'89705c4fe0f2cc5dccc9bedf58b46a903dc5eadf'}}

The output from diff
{'added': [(25, 'import org.junit.Ignore;'), (122, '\t@Ignore @Test')], 'deleted': [(121, '\t@Test')]}
'''


#https://stackoverflow.com/questions/70231772/how-to-get-last-touched-lines-in-a-file-of-a-commit-pydriller
#https://pydriller.readthedocs.io/en/latest/reference.html
#https://pydriller.readthedocs.io/en/latest/git.htmltomcat
