import pandas as pd
import math

'''
For irjit: Run python3 compute_metrics_at_each_timestamp.py >>  all_irjit_results_on_sampled_commits.csv
'''

def compute_confusion_matrix(data):
	tp=0
	fp=0
	tn=0
	fn=0

	commit = data['commit_id']
	actual = data['actual']       #actual = data.iloc[:,1]
	pred = data['predicted'] 	  #pred = data.iloc[:,2]

	for i in range(len(actual)):
		if (actual[i]==True and pred[i]==True):
			tp=tp+1
		elif(actual[i]==False and pred[i]==False):
			tn=tn+1
		elif(actual[i]==True and pred[i]==False):
			fn=fn+1
		elif(actual[i]==False and pred[i]==True):
			fp=fp+1
	return tn, fp, fn, tp

def compute_metrics(tn,fp,fn,tp):
	prec=0
	if (tp+fp)!=0:
		prec=tp/(tp+fp)
	rec=0
	if (tp+fn)!=0:
		rec= tp/(tp+fn)

	f1=0
	if prec!=0 or rec!=0:
		f1 = 2*((prec*rec)/(prec+rec))

	FAR = fp/(fp+tn)

	recall_1 = rec
	recall_0 = 1-FAR
	gmean = (recall_1 * recall_0) ** 0.5

	dist_heaven = math.sqrt((pow(1-rec,2)+pow(0-FAR,2))/2.0)

	return prec,recall_1,f1,gmean,FAR,dist_heaven

def main():
	group_size = 100
	start_idx = 0
	project = "neutron" #change manually
	K=3		    #change manually
	file1= "/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results/results_with_context_diff/reevaluate_resultlist_neutron_K=3.csv"
	#file1="/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results/results_lines_added_camel/resultlist_"+project+"_K="+str(K)+".csv"
	df = pd.read_csv(file1)

	#dupes = df[df.duplicated(subset='commit_id')]
	#print("IRJIT1:",len(df),len(dupes))
	TP=0
	TN=0
	FP=0
	FN=0
	print("project, tp, fp, fn, tn, total, prec,recall,f1,gmean,FAR,d2h, K, cumulative")
	while start_idx < len(df):
		end_idx = start_idx + min( group_size, (len(df)-start_idx))
		group = df.iloc[start_idx:end_idx]

		group = group.reset_index(drop=True)
		tn, fp, fn, tp = compute_confusion_matrix(group.copy())
		total = tp+fp+fn+tn
		#print("tp:",tp,"fp:",fp,"fn:",fn,"tn:",tn,"Total:",(total))
		prec,recall,f1,gmean,FAR,d2h = compute_metrics(tn, fp, fn, tp)
		print(f'{project}, {tp},{fp},{fn},{tn}, {total}, {prec:.3f}, {recall:.3f}, {f1:.3f}, {gmean:.3f}, {FAR:.3f}, {d2h:.3f}, {K}, No')

		TP += tp
		FP += fp
		FN += fn
		TN += tn
		Total = TP+FP+FN+TN
		#print("TP:",TP,"FP:",FP,"FN:",FN,"TN:",TN,"Total:",(Total))
		prec,recall,f1,gmean,FAR,d2h = compute_metrics(TN, FP, FN, TP)
		print(f'{project}, {TP},{FP},{FN},{TN}, {Total}, {prec:.3f}, {recall:.3f}, {f1:.3f}, {gmean:.3f}, {FAR:.3f}, {d2h:.3f}, {K}, Yes')
		start_idx += group_size

if __name__ == "__main__":
	main()
