import pandas as pd
import numpy as np
import time, pickle, math, warnings, os, operator, dill
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler
from CSVData import *
from my_util import *

warnings.filterwarnings('ignore')

remove_python_common_tokens = True

NOT_BUG = 0
BUG_NOT_DISCOVERED_W_DAYS = 1
BUG_DISCOVERED_W_DAYS = 2
BUG_FOUND = 3
W=90


data_path = './data/'
model_path = './final_model/'
top_k_tokens = np.arange(10,201,10)
agg_methods = ['avg','median','sum']
max_str_len_list = 100

# since we don't want to use commit metrics in LIME
#commit_metrics = ['la','la','ld', 'ld', 'nf','nd_y', 'nd', 'ns','ent', 'ent', 'nrev', 'rtime', 'hcmt', 'self', 'ndev',
                          #'age', 'age', 'nuc', 'app_y', 'aexp', 'rexp', 'arexp', 'rrexp', 'asexp', 'rsexp', 'asawr', 'rsawr']

commit_metrics = ['ns','nd','nf','entrophy','la','ld','ndev','age','nuc','exp','rexp','sexp']

line_score_df_col_name = ['total_tokens', 'line_level_label'] + ['token'+str(i) for i in range(1,max_str_len_list+1)] + [agg+'-top-'+str(k)+'-tokens' for agg in agg_methods for k in top_k_tokens] + [agg+'-all-tokens' for agg in agg_methods]

create_path_if_not_exist('./text_metric_line_eval_result/')
create_path_if_not_exist('./final_model/')
create_path_if_not_exist('./data/line-level_ranking_result/')


def get_combined_features(code_commit, commit_id, label, metrics_df, count_vect, mode = 'train'):

    if mode not in ['train','test']:
        print('wrong mode')
        return
    #print(commit_id)
    code_df = pd.DataFrame()
    code_df['commit_id'] = commit_id
    code_df['code'] = code_commit
    code_df['label'] = label

    code_df = code_df.sort_values(by='commit_id')

    metrics_df = metrics_df.sort_values(by='commit_id')

    code_change_arr = count_vect.transform(code_df['code']).astype(np.int16).toarray()

    if mode == 'train':
        metrics_df = metrics_df.drop('commit_id',axis=1)
        metrics_df_arr = metrics_df.to_numpy(dtype=np.float32)
        final_features = np.concatenate((code_change_arr,metrics_df_arr),axis=1)
        col_names = list(count_vect.get_feature_names())+list(metrics_df.columns)
        print("code_features:",len(code_df.columns))
        print("Tr metrics_df:",len(metrics_df.columns))
        return final_features, col_names, list(code_df['label'])
    elif mode == 'test':
        code_features = pd.DataFrame(code_change_arr, columns=count_vect.get_feature_names())
        code_features['commit_id'] = list(code_df['commit_id'])

        metrics_df = metrics_df.set_index('commit_id')
        code_features = code_features.set_index('commit_id')
        col_names = list(count_vect.get_feature_names())+list(metrics_df.columns)
        print("code_features:",len(code_features.columns))
        print("Test metrics_df:",len(metrics_df.columns))

        final_features = pd.concat([code_features, metrics_df],axis=1)
        return final_features, list(code_df['commit_id']), list(code_df['label'])


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
    commits_by_set = []
    commits=[]

    start_row = 0
    #This is to repeat the evaluation with same subsets 10 times. The subsets were randomly generated first time.
    #indexes_list_tomcat = [663,1525,2098,6028,6703,8177,8312,11612,15378,15769,22865,24042]
    #indexes_list_broadleaf  = [693,1723,4328,4774,5221,11035,11359,13424,14833,16191,17059,17428]
    #indexes_list_neutron = [2509, 3009, 3772, 5956, 10895, 12660, 15649, 16622, 17637, 18291, 22909, 24056]

    #indexes_list_spring = [898, 1817, 2505, 2841, 2977, 3144, 3642, 4361, 5432, 6882, 10926, 10998]
    #indexes_list_fabric8 =   [1381, 8223, 10797, 11063, 11723, 11729, 11898, 12365, 13241, 13526, 15452, 15577]
    #indexes_list_JGroups = [2536, 2538, 3436, 5840, 6074, 11122, 12763, 13647, 15349, 19256, 19559, 21467]

    #indexes_list_nova = [1666, 4027, 4994, 5493, 5503, 7523, 33388, 34781, 44627, 46894, 56997, 61361]
    #indexes_list_camel = [728, 3750, 3986, 10542, 20926, 20954, 26444, 26528, 27456, 29756, 35132, 36752]

    
    #indexes_list_brackets=[2210,6836, 12032,14634, 14874, 15036, 16272, 16614, 18360, 18708, 20006, 21345]
    indexes_list_npm = [959, 1018, 1793, 2966, 3599, 3720, 5499, 7220, 7846, 8113, 8861, 9257]

    for index, row in df.iterrows():
        commit_id = row['commit_id']
        commits.append(row.tolist())
        if index in indexes_list_npm:       	   #Make a change
            commits_by_set.append(commits.copy()) #.copy() ensures original commits do not change
    return commits_by_set


def eval_line_level(cur_proj, best_k_neighbor, df, train_data, test_data, iteration, CLH_queue, type3_dict):
    test_timestamp = test_data[0][15]
    train_data     = check_queue_inplace(train_data,CLH_queue,test_timestamp,type3_dict)
    train_code, train_commit, train_label = prepare_data(cur_proj, train_data, mode='train', remove_python_common_tokens=True)
    test_code, test_commit, test_label = prepare_data(cur_proj, test_data, mode='test',remove_python_common_tokens=True)
    print("Tr:",len(train_commit),"Test:",len(test_commit))
    print("Len Tr code:",len(train_code))

    commit_metrics = load_change_metrics_df(df)
    print("Len change metrics:",len(commit_metrics))
    train_commit_metrics = commit_metrics[commit_metrics['commit_id'].isin(train_commit)]
    test_commit_metrics = commit_metrics[commit_metrics['commit_id'].isin(test_commit)]
    # at this point we shall have correct train set to update dictionary
    count_vect = CountVectorizer(min_df=3, ngram_range=(1,1))
    count_vect.fit(train_code)

    train_feature, col_names, new_train_label    = get_combined_features(train_code, train_commit, train_label, train_commit_metrics, count_vect)
    test_feature, test_commit_id, new_test_label = get_combined_features(test_code,  test_commit,  test_label,  test_commit_metrics,  count_vect, mode = 'test')

    percent_80 = int(len(new_train_label)*0.8)
    final_train_feature = train_feature[:percent_80]
    final_new_train_label = new_train_label[:percent_80]
    print('load data of',cur_proj, 'finish')
    print("** ",type(final_train_feature),len(final_train_feature)) #numpy.ndarray of len =
    smote = SMOTE(k_neighbors = best_k_neighbor, random_state=42, n_jobs=-1)
    train_feature_res, new_train_label_res = smote.fit_resample(final_train_feature, final_new_train_label)
    print('resample data complete')
    fname = model_path+cur_proj+'_RF_DE_SMOTE_min_df_3'+'_'+str(iteration)+'.pkl'
    print(fname)
    clf = pickle.load(open(fname,'rb'))

    explainer = get_LIME_explainer(cur_proj, train_feature_res, col_names, iteration)
    print('load LIME explainer complete')
    del smote, train_feature_res, new_train_label_res, train_code, train_commit, train_label, test_code, test_commit, test_label
    del commit_metrics, train_commit_metrics, test_commit_metrics, count_vect, final_train_feature, final_new_train_label

    line_level_result = eval_with_LIME(cur_proj, clf, explainer, test_feature, iteration)
    if len(line_level_result) > 0:
        file_path = './data/'+cur_proj+'_line_level_result_min_df_3_300_trees'+str(iteration)+'.csv'
        pd.concat(line_level_result).to_csv(file_path,index=False,header=True)
        print('eval line level finish')
    return line_level_result



def get_LIME_explainer(proj_name, train_feature, feature_names, iteration):
    LIME_explainer_path = './final_model/'+proj_name+'_LIME_RF_DE_SMOTE_min_df_3_iteration_'+str(iteration)+'.pkl'
    class_names = ['not defective', 'defective'] # this is fine...
    if not os.path.exists(LIME_explainer_path):
        start = time.time()
        # get features in train_df here
        print('start training LIME explainer')

        explainer = LimeTabularExplainer(train_feature,
                                         feature_names=feature_names,
                                         class_names=class_names, discretize_continuous=False, random_state=42)

        dill.dump(explainer, open(LIME_explainer_path, 'wb'))
        print('finish training LIME explainer in',time.time()-start, 'secs')

    else:
        explainer = dill.load(open(LIME_explainer_path, 'rb'))

    return explainer


def eval_with_LIME(proj_name, clf, explainer, test_features, iteration):
    def preprocess_feature_from_explainer(exp):

        features_val = exp.as_list(label=1)

        new_features_val = [tup for tup in features_val if float(tup[1]) > 0] # only score > 0 that indicates buggy token

        feature_dict = {re.sub('\s.*','',val[0]):val[1] for val in new_features_val}

        sorted_feature_dict = sorted(feature_dict.items(), key=operator.itemgetter(1), reverse=True)

        sorted_feature_dict = {tup[0]:tup[1] for tup in sorted_feature_dict if tup[0] not in commit_metrics}

        tokens_list = list(sorted_feature_dict.keys())

        return sorted_feature_dict, tokens_list

    def add_agg_scr_to_list(line_stuff, scr_list):
        if len(scr_list) < 1:
            scr_list.append(0)

        line_stuff.append(np.mean(scr_list))
        line_stuff.append(np.median(scr_list))
        line_stuff.append(np.sum(scr_list))

    all_buggy_line_result_df = []
    #start_row_number_to_read = 1 + (iteration - 1) *100
    #end_row_number_to_read = start_row_number_to_read + 100
   
    line_level_df = pd.read_csv(data_path+proj_name+'_complete_buggy_line_level.csv',sep=',').dropna()

    start_row_number_to_read = (iteration - 1) *100
    end_row_number_to_read = start_row_number_to_read + 100
    df = pd.read_csv(data_path+proj_name+'_RF_DE_SMOTE_min_df_3_prediction_result.csv')
    prediction_result = df.iloc[start_row_number_to_read:end_row_number_to_read ]


    correctly_predicted_commit = list(prediction_result[(prediction_result['pred']==1) &
                                                (prediction_result['actual']==1)]['commit_id'])

    for commit in correctly_predicted_commit:
        print(commit)
        code_change_from_line_level_df = list(line_level_df[line_level_df['commit_hash']==commit]['code_change_remove_common_tokens'])
       
        if len(code_change_from_line_level_df) > 0:
            line_level_label = list(line_level_df[line_level_df['commit_hash']==commit]['is_buggy_line'])
            #print("line_level_label:",line_level_label)
            line_score_df = pd.DataFrame(columns = line_score_df_col_name)
            line_score_df['line_num'] = np.arange(0,len(code_change_from_line_level_df))
            line_score_df = line_score_df.set_index('line_num')
           
            try:
                exp = explainer.explain_instance(test_features.loc[commit], clf.predict_proba,
        		                                 num_features=len(test_features.columns), top_labels=1,
        		                                 num_samples=5000)

            except Exception as e:
    	        print(f"An error occurred: {str(e)}")

            sorted_feature_score_dict, tokens_list = preprocess_feature_from_explainer(exp)
            for line_num, line in enumerate(code_change_from_line_level_df): # for each line (sadly this loop is needed...)
                #print(line_num,line)
                line_stuff = []
                line_score_list = np.zeros(100) # this is needed to store result in dataframe
                token_list = line.split()[:100]
                line_stuff.append(line)
                line_stuff.append(len(token_list))

                for tok_idx, tok in enumerate(token_list):
                    score = sorted_feature_score_dict.get(tok,0)
                    line_score_list[tok_idx] = score

                # calculate top-k tokens first then followed by all tokens

                line_stuff = line_stuff + list(line_score_list)

                #print("line_stuff:",len(line_stuff),line_stuff)
                for k in top_k_tokens: # for each k in top-k tokens
                    top_tokens = tokens_list[0:k-1]
                    top_k_scr_list = []

                    if len(token_list) < 1:
                        top_k_scr_list.append(0)
                    else:
                        for tok in token_list:
                            score = 0
                            if tok in top_tokens:
                                score = sorted_feature_score_dict.get(tok,0)
                            top_k_scr_list.append(score)

                    add_agg_scr_to_list(line_stuff, top_k_scr_list)

                add_agg_scr_to_list(line_stuff, list(line_score_list[:len(token_list)]))
                line_score_df.loc[line_num] = line_stuff    #error here

            line_score_df['commit_id'] = [commit]*len(line_level_label)
            line_score_df['line_level_label'] = line_level_label
            #print(line_score_df)
            all_buggy_line_result_df.append(line_score_df)

            del exp, sorted_feature_score_dict, tokens_list, line_score_df

    return all_buggy_line_result_df


#  Defective line ranking evaluation
def create_tmp_df(all_commits,agg_methods):
    df = pd.DataFrame(columns = ['commit_id']+agg_methods)
    df['commit_id'] = all_commits
    df = df.set_index('commit_id')
    return df

def get_line_level_metrics(line_score,label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1)) # cannot pass line_score as list T-T
    pred = np.round(line_score)

    line_df = pd.DataFrame()
    line_df['scr'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr',ascending=False)
    line_df['row'] = np.arange(1, len(line_df)+1)

    real_buggy_lines = line_df[line_df['label'] == 1]
    #print(real_buggy_lines)
	
    top_10_acc = 0

    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2*len(line_df))

    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row']-1
        label_list = list(line_df['label'])

        all_rows = len(label_list)

        # find top-10 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows])/len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10])/len(label_list[:10])

        # find recall
        LOC_20_percent = line_df.head(int(0.2*len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num))/float(len(real_buggy_lines))

        # find effort @20% LOC recall

        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc

def eval_line_level_at_commit(cur_proj,iteration):
    score_cols = [agg+'-top-'+str(k)+'-tokens' for agg in agg_methods for k in top_k_tokens] + [agg+'-all-tokens' for agg in agg_methods]
    line_score_df_col_name = ['commit_id', 'total_tokens', 'line_level_label'] + score_cols


    RF_result = pd.read_csv(data_path+cur_proj+'_line_level_result_min_df_3_300_trees'+str(iteration)+'.csv')
    RF_result = RF_result[line_score_df_col_name]

    all_commits = list(RF_result['commit_id'].unique())

    IFA_df = create_tmp_df(all_commits, score_cols)
    recall_20_percent_effort_df = create_tmp_df(all_commits, score_cols)
    effort_20_percent_recall_df = create_tmp_df(all_commits, score_cols)
    precision_df = create_tmp_df(all_commits, score_cols)
    recall_df = create_tmp_df(all_commits, score_cols)
    f1_df = create_tmp_df(all_commits, score_cols)
    AUC_df = create_tmp_df(all_commits, score_cols)
    top_10_acc_df = create_tmp_df(all_commits, score_cols)
    MCC_df = create_tmp_df(all_commits, score_cols)
    bal_ACC_df = create_tmp_df(all_commits, score_cols)

    for commit in all_commits:
        IFA_list = []
        recall_20_percent_effort_list = []
        effort_20_percent_recall_list = []
        top_10_acc_list = []

        cur_RF_result = RF_result[RF_result['commit_id']==commit]

        to_save_df = cur_RF_result[['commit_id',  'total_tokens',  'line_level_label',  'sum-all-tokens']]

        scaler = MinMaxScaler()
        line_score = scaler.fit_transform(np.array(to_save_df['sum-all-tokens']).reshape(-1, 1))
        to_save_df['line_score'] = line_score.reshape(-1,1) # to remove [...] in numpy array
        to_save_df = to_save_df.drop(['sum-all-tokens','commit_id'], axis=1)
        to_save_df = to_save_df.sort_values(by='line_score', ascending=False)
        to_save_df['row'] = np.arange(1,len(to_save_df)+1)
        #===to_save_df.to_csv('./data/line-level_ranking_result/'+cur_proj+'_'+str(commit)+'.csv',index=False)

        line_label = list(cur_RF_result['line_level_label'])

        for n, agg_method in enumerate(score_cols):

            RF_line_scr = list(cur_RF_result[agg_method])
		
            IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc = get_line_level_metrics(RF_line_scr, line_label)

            IFA_list.append(IFA)
            recall_20_percent_effort_list.append(top_20_percent_LOC_recall)
            effort_20_percent_recall_list.append(effort_at_20_percent_LOC_recall)
            top_10_acc_list.append(top_10_acc)

        IFA_df.loc[commit] = IFA_list
        recall_20_percent_effort_df.loc[commit] = recall_20_percent_effort_list
        effort_20_percent_recall_df.loc[commit] = effort_20_percent_recall_list
        top_10_acc_df.loc[commit] = top_10_acc_list

    # the results are then used to make boxplot
    IFA_df.to_csv('./text_metric_line_eval_result/'+cur_proj+'_IFA_min_df_3_300_trees.csv', header = False, mode = 'a')
    recall_20_percent_effort_df.to_csv('./text_metric_line_eval_result/'+cur_proj+'_recall_20_percent_effort_min_df_3_300_trees.csv',header = False, mode = 'a',)
    effort_20_percent_recall_df.to_csv('./text_metric_line_eval_result/'+cur_proj+'_effort_20_percent_recall_min_df_3_300_trees.csv',header = False, mode = 'a',)
    top_10_acc_df.to_csv('./text_metric_line_eval_result/'+cur_proj+'_top_10_acc_min_df_3_300_trees.csv',header = False, mode = 'a',)




def run_experiment(cur_proj,number_of_splits,eval_size):
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
    #Copied from RQ1 so we don't re-tune using DE
    k_values_broadleaf = [15,13,19,19,10,17,13,18,13,18,14,18]
    k_values_spring = [19,20,19,6,6,9,12,13,19,20,15,15]
    k_values_fabric = [19,19,18,12,15,19,13,15,18,15,12,13]
    k_values_tomcat = [8,13,14,18,16,17,14,20,20,20,18,14]
    k_values_neutron = [15,4,4,11,4,16,18,19,19,10,14,13]
    k_values_JGroups = [11,17,19,6,6,12,12,6,20,11,7,4]
    k_values_nova = [8,19,16,6,13,4,18,15,20,20,20,16,18]
    k_values_camel = [2,10,9,18,15,8,15,10,19,18,17,19]
    k_values_npm = [14,6,14,9,17,18,13,17,15,16,14,17]
    k_values_brackets = [7,19,9,13,19,7,9,16,19,15,11,14]
    iteration = 0
    CLH_queue = []
    for eval_set in evaluation_set:

        iteration+=1

        trlen = len(eval_set)- eval_size
        train_data = eval_set[0:trlen]                          #all rows except last is for Tr
        test_data = eval_set[trlen:len(eval_set)]               #last row containing 18 columns
        print("==>Tr:",len(train_data),"Test:",len(test_data))
        test_df = pd.DataFrame(test_data)

        eval_line_level_at_commit(cur_proj,iteration)
        print('='*95)


def main():
    project = 'brackets'
    print(project)
    create_path_if_not_exist('./data/')
    create_path_if_not_exist('./final_model/')
    run_experiment(project,10,100)
    print('finish', project)


if __name__=="__main__":
    main()
