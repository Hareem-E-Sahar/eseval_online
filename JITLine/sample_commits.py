import csv,random,math
import pandas as pd

'''
def get_type3_commits_dictionary(df):
	#df = pd.read_csv(filename)
	#for a commit_id (key), the bug was reported at author_date_unix_timestamp (value)
	type3_data_rows = df[df['commit_type'] == 3]
	other = df[df['commit_type'] != 3]
	type3_dict = {}
	for i, row1 in type3_data_rows.iterrows():
		for j, row2 in other.iterrows():
			print(i,j)
			if rows_equal(row1,row2):
				author_date_unix_timestamp = row1['author_date_unix_timestamp']
				commit_id = row2['commit_id']
				type3_dict[commit_id]= author_date_unix_timestamp
	return type3_dict
'''

def get_type3_commits_dictionary(df):
	#for a commit_id (key), the bug was reported at author_date_unix_timestamp (value)
	type3_data_rows = df[df['commit_type'] == 3]
	other = df[df['commit_type'] != 3]
	other = other.drop_duplicates(subset=['commit_id']) 

	# Merge DataFrames based on rows_equal condition
	merged_df = pd.merge(other, type3_data_rows, how='inner', on=['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp', 'contains_bug'])

	# Rename the merged columns to match the original DataFrame
	merged_df = merged_df.rename(columns={'commit_id_x': 'commit_id','author_date_unix_timestamp_y': 'author_date_unix_timestamp'})


	# Create the dictionary using commit_id's from other and timestamp's from type3
	type3_dict = dict(zip(merged_df['commit_id'], merged_df['author_date_unix_timestamp']))

	return type3_dict


def rows_equal(row1,row2):
	return (
	row1['fix'] == row2['fix']
	and row1['ns'] == row2['ns']
	and row1['nd']== row2['nd']
	and row1['nf'] == row2['nf']
	and row1['entrophy'] == row2['entrophy']
	and row1['la']== row2['la']
	and row1['ld'] == row2['ld']
	and row1['lt'] == row2['lt']
	and row1['ndev']==row2['ndev']
	and row1['age'] == row2['age']
	and row1['nuc']== row2['nuc']
	and row1['exp'] == row2['exp']
	and row1['rexp']== row2['rexp']
	and row1['sexp']== row2['sexp']
	and row1['contains_bug'] == row2['contains_bug']
	)
	return False




def split_data_into_multiple_eval_sets(df,number_of_splits):
	print("split_data_into_multiple_eval_sets")
	split_index = math.ceil(int(0.1 * len(df))) 	           #initial 10%
	sampled_at_10_percent = df.iloc[split_index]['commit_id'] #manually sampled at 10%
	sampled_commits = sample_from_both_classes(df,number_of_splits)
	#sampled_commits = random.sample(df['commit_id'].tolist(), k=number_of_splits)
	sampled_commits.append(sampled_at_10_percent)

	sets = []
	commits_by_set = []
	commits=[]
	start_row = 0

	for index, row in df.iterrows():
		commit_id = row['commit_id']
		commits.append(row.tolist())

		if commit_id in sampled_commits:
			sampled_commits.remove(commit_id)
			commits_by_set.append(commits.copy()) #.copy() ensures previously appended commits in commits_by_set do not change
			sets.append((start_row, index))

	commits_by_set.append(commits) #after this point it won't be modified so passing original is Ok
	sets.append((start_row, index))
	print(sets)
	print(len(commits_by_set))
	return commits_by_set


def get_columns_with_missing_values(input_file):
	df = pd.read_csv(input_file)
	columns_with_missing_values = df.columns[df.isna().any()].tolist()
	print(columns_with_missing_values)

def get_duplicates(input_file):
	df = pd.read_csv(input_file)
	# Create a boolean mask to identify duplicate rows based on 'commit_id values
	duplicate_mask = df.duplicated(subset=['commit_id'], keep=False)
	duplicate_rows = df[duplicate_mask]
	duplicate_rows.to_csv('duplicate_rows.csv', index=False)# Filter and display the duplicate rows



# Now commits_by_set contains lists of commits for each set,
# sets contain row ranges for each set
input_file="/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/spring-integration_metrics.csv"
#commits_by_set = split_data_into_multiple_eval_sets (input_file,3)


