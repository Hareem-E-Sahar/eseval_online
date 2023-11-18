import re,json

"""
You can use json.dumps() to convert the explanation object into a JSON formatted string. 
However, the explanation object is a complex object with nested attributes, so might not 
be directly serializable to JSON, hence the ComplexEncoder.
Now ComplexEncoder converts explanation (python dict) into json string.
"""
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


class Explanation:
	def __init__(self):
		self.commit_tokens = []

		
	def get_explanation_tokens(self):
		if self.commit_tokens:
			self.commit_tokens.sort(key=lambda x: len(x)) #sort tokens asc by length

		return self.commit_tokens
		
	
	def print_tokens(self):
		print(self.get_explanation_tokens())	
		
		
	def extract_tokens_from_explanation(self,content):
		regex = r'weight\(lines_added:(\S+)'  # capture the token after 'lines_added:'
		pattern = re.compile(regex, re.MULTILINE)
		matcher = pattern.finditer(content)
		for match in matcher:
			self.commit_tokens.append(match.group(1))
		

	    

	def parse_explanation(self,data):
		explanation_json = json.dumps(data,cls=ComplexEncoder)
		data = json.loads(explanation_json)

		if isinstance(data, dict):
			if 'value' in data:
				pass
				#print(f"Value: {data['value']}")

			if 'description' in data:
				self.extract_tokens_from_explanation(data['description'])
				
			# Check if the dictionary has a 'details' key with a list value
			if 'details' in data and isinstance(data['details'], list):
				# Recursively process each item in the 'details' list
				for item in data['details']:
					self.parse_explanation(item)
	    
	   
