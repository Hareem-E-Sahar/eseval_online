from asyncio import sleep
from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from explain import * 

import json
import pandas as pd

def get_index_settings(filename):
    with open(filename) as f:
        index_settings = f.read()
        return index_settings

class ElasticsearchHandler:
    def __init__(self, host='localhost', port=9200):
        self.client = Elasticsearch(hosts=[{'host': host, 'port': port}])
        
    def check_health(self):
        response = self.client.cluster.health()
        return response

    def create_index(self,index_name,filename):
        custom_settings = get_index_settings(filename)
        response = self.client.indices.create(index=index_name,body=custom_settings)
        return response

    def index_json_document(self, json_document, index_name, doc_id):
        response = self.client.index(index=index_name,id=doc_id,body=json_document,doc_type='_doc')
        return response

    def index_bulk(self,bulk_data):
        #print(json.dumps(bulk_data, indent=2))
        try:
            success,failed = bulk(self.client,bulk_data,refresh=True)  # provide the bulk data
            if failed:
                for item in failed:
                    print(f"Failed operation: {item['index']['_id']}, reason: {item['index']['error']['reason']}")

        except Exception as e:
            print("Bulk request failed:", str(e))

    def delete_index(self,index_name):
        response = self.client.indices.delete(index=index_name, ignore=[400, 404])
        if response and response.get("acknowledged"):
            print("Index %s deleted successfully", str(index_name))
        else:
            print("Failed to delete index '{index_name}'")
        return response


class MoreLikeThisQuery:
    def __init__(self, index_name, host='localhost', port=9200):
        self.client = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = index_name
        self.similar_documents = []
        self.exp_obj = Explanation()

    def get_docs_from_hits(self, result):
        for hit in result:
            self.similar_documents.append({
                "doc_id": hit.meta.id,
                "score": hit.meta.score,
                #"commit":hit.to_dict()['commit_id'],
                #"filename":hit.to_dict()['filename'],
                "label": hit.to_dict()['buggy'],
                #"commit_filename": hit.to_dict()['commit_id']+"_"+os.path.basename(hit.to_dict()['filename'])
            })
            self.exp_obj.parse_explanation(hit.meta.explanation)    #self.exp_obj will have tokens from all queries of a commit
        self.similar_documents = sort_by_score(self.similar_documents)


    def execute_mlt_query(self, like_text, field, min_term_freq=1, min_doc_freq=1):
        s = Search(using=self.client, index=self.index_name)

        # Build the MLT query dynamically
        mlt_query = Q("more_like_this", fields=[field], like=like_text, min_term_freq=min_term_freq, min_doc_freq=min_doc_freq)

        s = s.query(mlt_query)  # add to search obj
        
        result = s.params(explain=True).execute()

        #result = s.execute()   # execute

        return self.get_docs_from_hits(result)


    def execute_mlt_query_bool(self, like_text, field, like_text2, field2, min_term_freq=1, min_doc_freq=1):
        s = Search(using=self.client, index=self.index_name)

        # Build the MLT query dynamically
        mlt_query1 = Q("more_like_this", fields=[field], like=like_text, min_term_freq=min_term_freq, min_doc_freq=min_doc_freq)
        mlt_query2 = Q("more_like_this", fields=[field2], like=like_text2, min_term_freq=min_term_freq, min_doc_freq=min_doc_freq)

        mlt_query = mlt_query1 | mlt_query2

        s = s.query(mlt_query)
        result = s.execute()

        return self.get_docs_from_hits(result)


def sort_by_score(data):
    sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)
    return sorted_data
