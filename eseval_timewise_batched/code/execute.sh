#!/bin/bash

python3 eseval_for_cabral_dataset_with_latency_v5.py -project nova -K 11 -settings "indexsettings_shingle.txt" -querytype "boolean"

python3 eseval_for_cabral_dataset_with_latency_v5.py -project spring-integration -K 3 -settings "indexsettings_shingle.txt" -querytype "boolean"
python3 eseval_for_cabral_dataset_with_latency_v5.py -project spring-integration -K 11 -settings "indexsettings_shingle.txt" -querytype "boolean"

python3 eseval_for_cabral_dataset_with_latency_v5.py -project JGroups -K 3 -settings "indexsettings_shingle.txt" -querytype "boolean"
python3 eseval_for_cabral_dataset_with_latency_v5.py -project JGroups -K 11 -settings "indexsettings_shingle.txt" -querytype "boolean"


python3 eseval_for_cabral_dataset_with_latency_v5.py -project tomcat -K 3 -settings "indexsettings_shingle.txt" -querytype "boolean"
python3 eseval_for_cabral_dataset_with_latency_v5.py -project tomcat -K 11 -settings "indexsettings_shingle.txt" -querytype "boolean"

python3 eseval_for_cabral_dataset_with_latency_v5.py -project fabric8 -K 3 -settings "indexsettings_shingle.txt" -querytype "boolean"
python3 eseval_for_cabral_dataset_with_latency_v5.py -project fabric8 -K 11 -settings "indexsettings_shingle.txt" -querytype "boolean"
