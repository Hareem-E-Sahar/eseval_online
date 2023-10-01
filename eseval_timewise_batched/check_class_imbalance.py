import pandas as pd
import math

'''
For irjit: check class imbalance
'''

def compute_total_buggy(data):
	commit = data['commit_id']
	actual = data['actual']       #actual = data.iloc[:,1]

	buggy=0
	clean=0
	for i in range(len(actual)):
		if (actual[i]==True):
			buggy+=1
		elif(actual[i]==False):
			clean+=1
	return buggy,clean

def main():
	group_size = 100
	start_idx = 0
	project = "neutron" #change manually
	K=3 #change manually
	file1="/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results/results_lines_added_camel/resultlist_"+project+"_K="+str(K)+".csv"
	df = pd.read_csv(file1)

	dupes = df[df.duplicated(subset='commit_id')]
	print("IRJIT1:",len(df),len(dupes))
	
	print("project, buggy, %buggy, clean, %clean")
	while start_idx < len(df):
		end_idx = start_idx + min( group_size, (len(df)-start_idx))
		group = df.iloc[start_idx:end_idx]
		group = group.reset_index(drop=True)
		buggy,clean = compute_total_buggy(group.copy())
		print(f'{project}, {buggy},{buggy/100},{clean},{clean/100}')
		start_idx += group_size

if __name__ == "__main__":
	main()
