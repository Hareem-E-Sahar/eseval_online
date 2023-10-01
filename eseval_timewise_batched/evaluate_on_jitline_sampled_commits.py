import pandas as pd
import math

'''Use only if you don't want to run irjit again on sampled_jitline commits'''
def compute_confusion_matrix(data, jitline_commits):
	tp=0
	fp=0
	tn=0
	fn=0
	print(type(jitline_commits))
	for i in range(len(data)):
		commit = data.iloc[i,0]
		actual = data.iloc[i,1] #actual = data.iloc[:,1]
		pred = data.iloc[i,2] 	  #pred = data.iloc[:,2]
		if commit in jitline_commits.tolist():

			if (actual==True and pred==True):
				tp=tp+1
			elif(actual==False and pred==False):
				tn=tn+1
			elif(actual==True and pred==False):
				fn=fn+1
			elif(actual==False and pred==True):
				fp=fp+1

		#print(actual[i], pred[i])
	return tn, fp, fn, tp

def compute_metrics(tn,fp,fn,tp):
	prec=tp/(tp+fp)
	print("prec:",prec)

	rec = tp/(tp+fn)
	print("rec:",rec)

	f1 = 2*((prec*rec)/(prec+rec))
	print("f1:",f1)

	FAR = fp/(fp+tn)

	recall_1 = rec
	recall_0 = 1-FAR
	gmean = (recall_1 * recall_0) ** 0.5
	print("gmean:",gmean)


	print("FAR:",FAR)



	dist_heaven = math.sqrt((pow(1-rec,2)+pow(0-FAR,2))/2.0)
	print("d2h:",dist_heaven)
	return prec,recall_1,f1,gmean,FAR,dist_heaven


project = "fabric8"
K=3
file1="/home/hareem/UofA2023/eseval_v2/eseval_timewise/results/resultlist_all_projects/resultlist_"+project+"_K="+str(K)+".csv"
data = pd.read_csv(file1)
file2="/home/hareem/UofA2023/JITLine-replication-package/JITLine/data/online_eval_results/sampled_test_commits/"+project+"_sampled_test_commits.csv"

df = pd.read_csv(file2)

dupes1 = df[df.duplicated(subset='commit_id')]
print("IRJIT1:",len(df),len(dupes1))


jitline_commits = df.iloc[:,17]

tn, fp, fn, tp = compute_confusion_matrix(data,jitline_commits)
print("tp:",tp,"fp:",fp,"fn:",fn,"tn:",tn)
print("Total:",tp+fp+fn+tn)
print('\n')

prec,recall,f1,gmean,FAR,d2h = compute_metrics(tn, fp, fn, tp)
print(prec,recall,f1,gmean,FAR,d2h)
