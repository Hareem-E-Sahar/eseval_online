# eseval_online
Bug prediction evaluation in an online fashion

## Step1: Clone the repo
The repository contains:
1. JITLine: This directory contains source code that we used conduct batched evaluation of baseline approach.
2. cabral_dataset: This contains dataset of Cabral et al. Now we also included commit ids to the original data. 
3. eseval_timewise: This folder contains code and the results of online evaluation of IRJIT.
4. eseval_timewise_batched: This folder contains code and the results of batched evaluation of IRJIT. This folder 
also contains the code to conduct line level evaluation, the code for which is in the linelevel_code directory.
5. linelevel_data: This folder has line level data and also the code used to extract line level data.  
6. plots: This folder contains the R scripts used to plot figures presented in the paper.


## Step2: Reproduce the results
1. Download elasticsearch latest version (https://www.elastic.co/downloads/elasticsearch) and run the server before running irjit evaluation.

2. To run the online evaluation for IRJIT go to eseval_timewise/code and run this command. \
python3 eseval_for_cabral_dataset_with_latency_v6.py -project npm -K 5\


3. To run the batched evaluation for IRJIT go to eseval_timewise_batched/code and run this command. \
python3 eseval_for_cabral_dataset_with_latency_v5.py -project camel -K 3 -settings "indexsettings_camel.txt" -querytype "notboolean"\


4. To run line level evaluation go to eseval_timewise_batched/linelevel_code and run this command. \
python3 eseval_linelevel.py -project brackets -K 3 -settings "indexsettings_camel.txt" -querytype "notboolean"\
The code uses data in the linelevel_data folder.


5. To run the batched evaluation for JITLine first set up the environment using following commands.\
conda env create --file requirements.yml\
conda activate JITLine go to JITLine \
Then run the following command
python3 JITLine_RQ1-RQ2.py\
python3 JITLine-RQ3.py\
for commit level and line level evaluation respectively.


## Step3: Reproduce the figures
1. Reproduce plots using the aggregate results and the scripts available in the plots directory.
