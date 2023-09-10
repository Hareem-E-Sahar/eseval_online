from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch

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

    def create_index(self,index_name):
        custom_settings = get_index_settings("/home/hareem/UofA2023/eseval_v2/eseval_timewise/code/indexsettings.txt")
        response = self.client.indices.create(index=index_name,body=custom_settings)
        return response

    def index_json_document(self, json_document, index_name, doc_id):
        response = self.client.index(index=index_name,id=doc_id,body=json_document,doc_type='_doc')
        return response

    def index_bulk(self,json_document,index_name,doc_id):
        try:
            response = self.client.bulk(index=index_name, body=bulk_data,refresh=True)  # Specify the index and provide the bulk data
            print("Bulk request executed successfully.")
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

    def execute_mlt_query(self, like_text, field, min_term_freq=1, min_doc_freq=1):
        s = Search(using=self.client, index=self.index_name)

        # Build the MLT query dynamically
        mlt_query = Q("more_like_this", fields=[field], like=like_text, min_term_freq=min_term_freq, min_doc_freq=min_doc_freq)
        s = s.query(mlt_query) # add to search obj
        result = s.execute()   # execute
        for hit in result:
           #if hit.meta.id != like_doc_id:
           self.similar_documents.append({
            "doc_id": hit.meta.id,
            "score": hit.meta.score,
            #"commit":hit.to_dict()['commit_id'],
            #"filename":hit.to_dict()['filename'],
            "label": hit.to_dict()['buggy'],
            #"commit_filename": hit.to_dict()['commit_id']+"_"+os.path.basename(hit.to_dict()['filename'])
        })
        self.similar_documents = sort_by_score(self.similar_documents)



def sort_by_score(data):
    sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)
    return sorted_data
