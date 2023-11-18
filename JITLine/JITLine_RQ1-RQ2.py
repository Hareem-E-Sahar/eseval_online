# import pickle
from my_util import *

import numpy as np
from timeit import default_timer as timer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, classification_report, auc
from imblearn.over_sampling import SMOTE

import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import time, pickle, math, warnings, os
from sample_commits import *
warnings.filterwarnings('ignore')


RF_data_dir = './data/'
sampling_methods = 'DE_SMOTE_min_df_3'
remove_python_common_tokens = True

NOT_BUG = 0
# the commit is buggy but its true label was not found within W days for test dataset
BUG_NOT_DISCOVERED_W_DAYS = 1
#the commit is buggy and its true label was found within W days for test dataset
BUG_DISCOVERED_W_DAYS = 2
#the true label of a defective commit was assigned.
BUG_FOUND = 3
W=90

def get_combined_df(code_commit, commit_id, label, metrics_df, count_vect):

    print("Len commit_id:",len(commit_id))
    code_df = pd.DataFrame()
    code_df['commit_id'] = commit_id
    code_df['code'] = code_commit
    code_df['label'] = label

    code_df = code_df.sort_values(by='commit_id')
    metrics_df = metrics_df.sort_values(by='commit_id')
    metrics_df = metrics_df.drop('commit_id',axis=1)
	
    code_change_arr = count_vect.transform(code_df['code']).astype(np.int16).toarray()
    metrics_df_arr = metrics_df.to_numpy(dtype=np.float32)

    final_features = np.concatenate((code_change_arr,metrics_df_arr),axis=1)
    print("final features:", final_features.shape[1]) #272
    return final_features, list(code_df['commit_id']), list(code_df['label'])


def calculate_gmean(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    sensitivity = tp / (tp + fn)  #recall_1
    specificity = tn / (tn + fp)  #recall_0
    gmean = (sensitivity * specificity) ** 0.5
    return gmean

def objective_func(k, train_feature, train_label, valid_feature, valid_label):
    smote = SMOTE(random_state=42, k_neighbors=int(np.round(k)), n_jobs=32)
    train_feature_res, train_label_res = smote.fit_resample(train_feature, train_label)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(train_feature_res, train_label_res)

    predictions = clf.predict(valid_feature)

    conf_matrix = confusion_matrix(valid_label, predictions)
    gmean = calculate_gmean(conf_matrix)
    return -gmean


def check_CLH_queue(train_data,CLH_queue,current_timestamp,type3_dict):
    for element in CLH_queue:
        tr_timestamp = element[15]
        commit_type= element[16]
        commit_id = element[17]
        if (defect_found_by_current_timestamp(commit_id, current_timestamp, type3_dict) is True):
            element[14] = True
            train_data.insert(0,element) #insert at start
            CLH_queue.remove(element)


def check_queue_inplace(WFL_queue,CLH_queue,current_timestamp,type3_dict):
    #change train_set inplace to remove data not suitable for training
    #returns updated Tr set
    for element in WFL_queue:
        tr_timestamp = element[15]
        commit_type= element[16]
        commit_id = element[17]

        if commit_type==NOT_BUG or commit_type==BUG_NOT_DISCOVERED_W_DAYS:#0, #1
            if calculate_time_elapsed(current_timestamp,tr_timestamp)>=W:
                element[14] = False       #its clean and will be kept for training
                CLH_queue.append(element) #is clean
                pass

            else:
                WFL_queue.remove(element)

        elif commit_type == BUG_DISCOVERED_W_DAYS:#2
            if (defect_found_by_current_timestamp(commit_id, current_timestamp, type3_dict) is False):
                #its bug has NOT been reported so you can't keep it in Tr
                WFL_queue.remove(element)
    return WFL_queue


def split_data_into_multiple_eval_sets(df,number_of_splits):

	print("split_data_into_multiple_eval_sets")

	'''
	split_index = math.ceil(int(0.1 * len(df))) 			  #initial 10%
	sampled_at_10_percent = df.iloc[split_index]['commit_id']        #manually sampled at 10%
	sampled_commits = sample_from_both_classes(df,number_of_splits)
	sampled_commits.append(sampled_at_10_percent)
	'''
	
	commits_by_set = []
	commits=[]
	start_row = 0

	#indexes_list_tomcat = [663,1525,2098,6028,6703,8177,8312,11612,15378,15769,22865,24042]
	indexes_list_neutron = [2509, 3009, 3772, 5956, 10895, 12660, 15649, 16622, 17637, 18291, 22909, 24056]
	#indexes_list_JGroups = [2536, 2538, 3436, 5840, 6074, 11122, 12763, 13647, 15349, 19256, 19559, 21467]
	
	#indexes_list_broadleaf  = [693,1723,4328,4774,5221,11035,11359,13424,14833,16191,17059,17428]
	#indexes_list_spring = [898, 1817, 2505, 2841, 2977, 3144, 3642, 4361, 5432, 6882, 10926, 10998]
	#indexes_list_fabric8 =   [1381, 8223, 10797, 11063, 11723, 11729, 11898, 12365, 13241, 13526, 15452, 15577]
	
	#indexes_list_nova = [1666, 4027, 4994, 5493, 5503, 7523, 33388, 34781, 44627, 46894, 56997, 61361]
	#indexes_list_camel = [728, 3750, 3986, 10542, 20926, 20954, 26444, 26528, 27456, 29756, 35132, 36752]
	#indexes_list_brackets=[2210,6836, 12032,14634, 14874, 15036, 16272, 16614, 18360, 18708, 20006, 21345]
	#indexes_list_npm = [959, 1018, 1793, 2966, 3599, 3720, 5499, 7220, 7846, 8113, 8861, 9257]
		

	for index, row in df.iterrows():
		commit_id = row['commit_id']
		commits.append(row.tolist())
		if index in indexes_list_neutron:
			commits_by_set.append(commits.copy()) #.copy() ensures original commits do not change
	return commits_by_set


def run_experiment(cur_proj,number_of_splits,eval_size):
    #sets = number_of_splits + 1 (or 2 for even)
    data_path = './data/'
    model_path = './final_model/'
    RF_data_dir = './data/'

    csv_file = RF_data_dir + 'change_metrics/'+ cur_proj + "_metrics.csv"
    
    df = pd.read_csv(csv_file)
    type3_dict = get_type3_commits_dictionary(df) #need original dataframe
    df = df.dropna()
    df = df.drop_duplicates(subset=['commit_id'])

    evaluation_set = split_data_into_multiple_eval_sets(df,number_of_splits)
    print("# of sets:",len(evaluation_set))

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    gmean_values = []
    WFL_queue = []
    CLH_queue = []
    iteration=0

    for eval_set in evaluation_set:
        iteration+=1
        #WFL_queue = train_data
        trlen = len(eval_set)- eval_size
        train_data = eval_set[0:trlen]                          #all rows except last is for Tr
        test_data  = eval_set[trlen:len(eval_set)]              #last row containing 18 columns
        print("==>Tr:",len(train_data),"Test:",len(test_data))

        test_df = pd.DataFrame(test_data)
        test_df.columns =['fix','ns','nd','nf','entrophy','la','ld','lt','ndev','age','nuc','exp','rexp','sexp','contains_bug','author_date_unix_timestamp','commit_type','commit_id']
        test_df.to_csv(data_path+cur_proj+"_sampled_test_commits.csv",mode='a',header=False,index=False)
        
        test_timestamp = test_data[0][15]                          #15=timestamp
        train_data     = check_queue_inplace(train_data,CLH_queue,test_timestamp,type3_dict)
	
       
        train_code, train_commit, train_label = prepare_data(cur_proj, train_data, mode='train', remove_python_common_tokens=True)
        start_test = time.time()
        test_code, test_commit, test_label    = prepare_data(cur_proj, test_data, mode='test',remove_python_common_tokens=True)
        print("Tr:",len(train_commit),"Test:",len(test_commit))
        
        	
        commit_metrics = load_change_metrics_df(df)
        print("Len change metrics:",len(commit_metrics))
        train_commit_metrics = commit_metrics[commit_metrics['commit_id'].isin(train_commit)]
        test_commit_metrics = commit_metrics[commit_metrics['commit_id'].isin(test_commit)]


        # at this point we shall have correct train set to update dictionary
        count_vect = CountVectorizer(min_df=3, ngram_range=(1,1))
        count_vect.fit(train_code)

        
        test_feature, test_commit_id, new_test_label = get_combined_df(test_code, test_commit, test_label, test_commit_metrics,count_vect)
        end_test = time.time()
        test_time = end_test-start_test
        
        train_feature, train_commit_id, new_train_label = get_combined_df(train_code, train_commit, train_label, train_commit_metrics,count_vect)
        
        #this splits data into training and validation
        percent_80 = int(len(new_train_label)*0.8)

        final_train_feature = train_feature[:percent_80]
        final_train_commit_id = train_commit_id[:percent_80]
        final_new_train_label = new_train_label[:percent_80]

        valid_feature = train_feature[percent_80:]
        valid_commit_id = train_commit_id[percent_80:]
        valid_label = new_train_label[percent_80:]

        print('load data of',cur_proj, 'finish')
        start = time.time()

        bounds = [(1,20)]
        
        result = differential_evolution(objective_func, bounds, args=(final_train_feature, final_new_train_label,
                                                                      valid_feature, valid_label),
                                       popsize=10, mutation=0.7, recombination=0.3,seed=0)
      	
      	
        
        print("k_neighbors:",int(np.round(result.x)))
        smote = SMOTE(random_state=42, n_jobs=32, k_neighbors=int(np.round(result.x)))
        train_feature_res, train_label_res = smote.fit_resample(final_train_feature, final_new_train_label)
        print("** ",type(train_feature_res),len(train_feature_res)) #numpy.ndarray of len = 294
              
        print("** ",type(train_label_res),len(train_label_res))     #list of labels with len = 294

        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf_name = 'RF'
        print("Test feature:",type(test_feature),test_feature.shape[1]) # len = 272

        trained_clf, pred_df, pred, actual = train_eval_model(clf, train_feature_res, train_label_res,
                                           test_feature, new_test_label)
        end = time.time()
        train_time = end-start
        print("Train time:",train_time)
        print("Test time:",test_time)
        print("# of test instances:",len(new_test_label))

        pred_df['test_commit'] = test_commit_id #commit_id for q4
        
        pred_df.to_csv(data_path+cur_proj+'_'+clf_name+'_'+sampling_methods+'_prediction_result.csv',mode='a', header=False, index=False)
       
        test_pred_df = pd.merge(test_df,pred_df, left_on=["commit_id"],right_on=["test_commit"])
        
        test_pred_df.to_csv(data_path+cur_proj+'_'+clf_name+'_'+sampling_methods+'_prediction_result_with_test_data_info.csv',mode='a', header=False, index=False)


        clf_file_path = model_path+cur_proj+'_'+clf_name+'_'+sampling_methods+'_'+str(iteration)+'.pkl'
        pickle.dump(trained_clf, open(clf_file_path, 'wb'))
       
        try:
            tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, 0  
	
        print("tn:",tn,"fp:",fp,"fn:",fn,"tp:",tp)
        TN = TN + tn
        FP = FP + fp
        FN = FN + fn
        TP = TP + tp
        print("TN:",TN,"FP:",FP,"FN:",FN,"TP:",TP)
        print('-'*100)
	

    print("g-mean values:", gmean_values)
    prec, rec, f1, gmean, FAR, dist_heaven = eval_metrics(TP, TN, FP, FN)
    print('Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, Gmean: {:.2f}, FAR: {:.2f}, d2h: {:.2f}'.format(prec, rec, f1, gmean, FAR, dist_heaven))


	
def eval_metrics(TP, TN, FP, FN):
    prec = TP/(TP+FP)
    recall_1 = TP/(TP+FN)
    f1 = 2*((prec * recall_1)/(prec + recall_1))
    FAR = FP/(FP+TN)
    dist_heaven = math.sqrt((pow(1- recall_1, 2)+pow(0-FAR,2))/2.0)
    recall_0 = 1-FAR
    gmean = (recall_1 * recall_0) ** 0.5
    return prec, recall_1, f1, gmean, FAR, dist_heaven



def main():

    project = 'neutron' 

    create_path_if_not_exist('./data/')
    create_path_if_not_exist('./final_model/')
    run_experiment(project,10,100)

if __name__=="__main__":
    main()
