from sklearn.preprocessing import MinMaxScaler
from pstats import SortKey
import pandas as pd
import numpy as np
import math


def save_linelevel_result(project,K,cm,results_dir,line_level_result):
    file_path = results_dir+project+'_line_level_result.csv'
    pd.concat(line_level_result).to_csv(file_path,index=False,header=True)
    print('eval line level finish')


def rank_ground_truth_lines(project, commit, commit_buggy_tokens_list):
    folder = "/home/hareem/UofA2023/eseval_v2/eseval_online/linelevel_data/"
    line_level_df = pd.read_csv(folder + project + '_complete_buggy_line_level.csv', sep=',' ).dropna()
    commit_df = line_level_df[line_level_df['commit_hash'] == commit]
    if commit_df.empty:
        return None

    else:
        
        lines_info = []  # Create a list of tuples with line, token count, and label
        for index, row in commit_df.iterrows():
            line = row['code_change_remove_common_tokens']
            line_tokens = line.split()
            label = row['is_buggy_line']
            count = sum(1 for token in line_tokens if token in commit_buggy_tokens_list)
            
            lines_info.append((line, count, label)) #  tuple 
        # Sort the list of tuples by token count (descending)
        sorted_lines_info = sorted(lines_info, key=lambda item: item[1], reverse=True)
        
        
        line_score_df = pd.DataFrame(columns = ['line','commit_id','count','line_level_label'])       
        line_score_df['line_num'] = np.arange(0,len(commit_df['code_change_remove_common_tokens']))
        line_score_df = line_score_df.set_index('line_num')
          
        line_score_df['line'] = sorted_lines = [info[0] for info in sorted_lines_info]
        line_score_df['commit_id'] = commit
        line_score_df['count'] = sorted_counts = [info[1] for info in sorted_lines_info]
        line_score_df['line_level_label'] =  sorted_labels = [info[2] for info in sorted_lines_info]
    
        #line_score_df.loc[index] =     #['line_value', commit, count, label, commit_buggy_tokens_list]

        #line_score_df['tokens'] = commit_buggy_tokens_list

        return line_score_df
    

def create_tmp_df(all_commits):
        df = pd.DataFrame(columns = ['commit_id']+['count']) #label and line left
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



def eval_line_level_at_commit(project, results_dir):
    #score_cols = [agg+'-top-'+str(k)+'-tokens' for agg in agg_methods for k in top_k_tokens] + [agg+'-all-tokens' for agg in agg_methods]
    #line_score_df_col_name = ['commit_id', 'total_tokens', 'line_level_label'] + score_cols
    #score_cols = [count]
    
    RF_result = pd.read_csv(results_dir+project+'_line_level_result.csv')
    print(type(RF_result))
    
    #RF_result = RF_result[line_score_df_col_name]
  
    all_commits = list(RF_result['commit_id'].unique())


    IFA_df = create_tmp_df(all_commits)
    precision_df = create_tmp_df(all_commits)
    recall_20_percent_effort_df = create_tmp_df(all_commits)
    precision_df = create_tmp_df(all_commits)
    effort_20_percent_recall_df = create_tmp_df(all_commits)
    precision_df = create_tmp_df(all_commits)
    recall_df = create_tmp_df(all_commits)
    f1_df = create_tmp_df(all_commits)
    AUC_df = create_tmp_df(all_commits)
    top_10_acc_df = create_tmp_df(all_commits)
    MCC_df = create_tmp_df(all_commits)
    bal_ACC_df = create_tmp_df(all_commits)

    for commit in all_commits:
        IFA_list = []
        recall_20_percent_effort_list = []
        effort_20_percent_recall_list = []
        top_10_acc_list = []

        cur_RF_result = RF_result[RF_result['commit_id']==commit]

        to_save_df = cur_RF_result[['commit_id',  'line',  'line_level_label',  'count']].copy()

        #to_save_df = cur_RF_result[['commit_id',  'line',  'line_level_label',  'count']]

        scaler = MinMaxScaler()
        line_score = scaler.fit_transform(np.array(to_save_df['count']).reshape(-1, 1))
        #to_save_df['line_score'] = line_score.reshape(-1,1) # to remove [...] in numpy array
        to_save_df.loc[:, 'line_score'] = line_score.reshape(-1, 1)

        #to_save_df = to_save_df.drop(['count','commit_id'], axis=1)
        
        to_save_df = to_save_df.sort_values(by='count', ascending=False)
        
        to_save_df['row'] = np.arange(1,len(to_save_df)+1)
        to_save_df.to_csv(results_dir+'line-level_ranking_result/'+project+'_'+str(commit)+'.csv',index=False)

        line_label = list(cur_RF_result['line_level_label'])
        
        
       # for n, agg_method in enumerate([score_cols]):

        RF_line_scr = list(cur_RF_result['count'])

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
    IFA_df.to_csv(results_dir+'text_metric_line_eval_result/'+project+'_IFA_min_df_3_300_trees.csv', header = False, mode = 'a')
    recall_20_percent_effort_df.to_csv(results_dir+'text_metric_line_eval_result/'+project+'_recall_20_percent_effort_min_df_3_300_trees.csv',header = False, mode = 'a',)
    effort_20_percent_recall_df.to_csv(results_dir+'text_metric_line_eval_result/'+project+'_effort_20_percent_recall_min_df_3_300_trees.csv',header = False, mode = 'a',)
    top_10_acc_df.to_csv(results_dir+'text_metric_line_eval_result/'+project+'_top_10_acc_min_df_3_300_trees.csv',header = False, mode = 'a',)




