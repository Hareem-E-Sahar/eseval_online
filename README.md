# eseval_online
**Bug Prediction Evaluation in an Online Fashion**

## Step 1: Clone the Repo
The repository contains several important directories:
- **JITLine**: Contains source code used to conduct batched evaluation of the baseline approach.
- **cabral_dataset**: Includes the dataset of Cabral et al., now with commit IDs added to the original data. Also contains a script 
to extract source code changes of each commit and save as json files.
- **eseval_timewise**: Contains code and results for the online evaluation of IRJIT.
- **eseval_timewise_batched**: Has code and results for the batched evaluation of IRJIT, including line-level evaluation code found in the `linelevel_code` directory.
- **linelevel_data**: Features line-level data and code used to extract this data.
- **plots**: Contains R scripts for plotting figures presented in the paper.

## Step 2: Reproduce the Results
Follow these steps to replicate the evaluation results:
1. **Set Up Elasticsearch**:
   - Download the latest version of Elasticsearch from [Elastic.co](https://www.elastic.co/downloads/elasticsearch).
   - Run the Elasticsearch server before executing IRJIT evaluations.

2. **Online Evaluation for IRJIT**:
   - Navigate to `eseval_timewise/code`.
   - Run the following command:
     ```
     python3 eseval_for_cabral_dataset_with_latency_v6.py -project npm -K 5
     ```
   - Before running the above command make sure you have obtained the source code changes for each commit and saved those as json files. A few sample json files are inside `~/cabral_dataset/npm/data/npm_jsonfiles`
3. **Batched Evaluation for IRJIT**:
   - Go to `eseval_timewise_batched/code`.
   - Execute this command:
     ```
     python3 eseval_for_cabral_dataset_with_latency_v5.py -project camel -K 3 -settings "indexsettings_camel.txt" -querytype "notboolean"
     ```

4. **Line Level Evaluation**:
   - Move to `eseval_timewise_batched/linelevel_code`.
   - Use this command:
     ```
     python3 eseval_linelevel.py -project brackets -K 3 -settings "indexsettings_camel.txt" -querytype "notboolean"
     ```
   - This code utilizes data in the `linelevel_data` folder.

5. **Batched Evaluation for JITLine**:
   - Set up the environment with these commands:
     ```
     conda env create --file requirements.yml
     conda activate JITLine
     ```
   - Then, in the JITLine directory, run:
     ```
     python3 JITLine_RQ1-RQ2.py
     python3 JITLine-RQ3.py
     ```
   - For commit level and line level evaluation, respectively.

## Step 3: Reproduce the Figures
1. Use the aggregate results and scripts available in the `plots` directory to reproduce the plots.

