import os
import shutil
docs_dir = "apwsj_parsed_documents/"
err_dir="topic_files_not_found.txt"
count=0
for line in open("topic_doc_mapping.txt","r"):
	tokens = line.split()
	topic = tokens[0]
	files = tokens[1:]
	absent_files=[]
	if not os.path.exists(docs_dir+topic+"/"):
		os.makedirs(docs_dir+topic+"/")
	for file in files:
		try:
			shutil.copy(docs_dir+file,docs_dir+topic+"/"+file)
		except:
			absent_files.append(file)
	with open(err_dir,"a") as f:
		f.write(" ".join([topic]+absent_files)+"\n")
	print(topic+" done")
	count+=len(absent_files)
print(count)
