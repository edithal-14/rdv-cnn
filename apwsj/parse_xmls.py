import os
import xml.etree.ElementTree as ET
import sys
dir="apwsj_documents/"
new_dir="apwsj_parsed_documents/"
error_file="parsing_errors.txt"
if not os.path.exists(new_dir):
	os.makedirs(new_dir)
files = os.listdir(dir)
counter=0
for file in files:
	lines = open(dir+file).read().splitlines()
	doc_tag_positions = [i for i in range(len(lines)) if lines[i]=="<DOC>"]
	documents = ["\n".join(lines[doc_tag_positions[i]:doc_tag_positions[i+1]]) for i in range(len(doc_tag_positions)-1)]
	documents.append("\n".join(lines[doc_tag_positions[-1]:]))
	for doc in documents:
		docid = None
		doctext= None
		try:
			root = ET.fromstring(doc)
			for node in root:
                        	if node.tag=="DOCNO":
                                	docid = node.text[1:-1]
                        	elif node.tag=="TEXT":
                                	if doctext:
                                        	doctext=doctext+"\n"+node.text
                                	else:
                                        	doctext = node.text
				elif node.tag=="LP":
					if doctext:
						doctext = doctext+"\n"+node.text
					else:
						doctext = node.text
		except Exception as e:
			#manual parsing required
			next=0
			copy=0
			doctext=""
			for line in doc.splitlines():
				if next==1:
					docid = line.strip()
					next=0
				if line.startswith("<DOCNO>"):
					if line=="<DOCNO>":
						next=1
					else:
						docid = line.split()[1]
				if line=="</TEXT>":
					copy=0
				if line=="</LP>":
					copy=0
				if copy==1:
					doctext = doctext+line+"\n"
				if line=="<TEXT>":
					copy=1
				if line=="<LP>":
					copy=1
					
		try:
			with open(new_dir+docid,"w") as f:
				f.write(doctext)
		except Exception as e:
			with open(error_file,"a") as f:
				f.write(str(e)+"\n"+doc+"\n\n")
	counter+=1
	if counter%50==0:
		print(str(counter)+" documents processed")
		
