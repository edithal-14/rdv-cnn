import os
from collections import defaultdict
topics_allowed="q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
topics_allowed=topics_allowed.split(", ")
write_file="novel_list_without_partially_redundant.txt"
topic_dir="apwsj_parsed_documents"
topics=os.listdir(topic_dir)
doc_topic_dict=defaultdict(list)
for topic in topics:
	for doc in os.listdir(topic_dir+"/"+topic):
		doc_topic_dict[doc].append(topic)
docs_sorted = open("NoveltyData/apwsj88-90.rel.docno.sorted","r").read().splitlines()
sorted_doc_topic_dict = defaultdict(list)
for doc in docs_sorted:
	if len(doc_topic_dict[doc])>0:
		for t in doc_topic_dict[doc]:
			sorted_doc_topic_dict[t].append(doc)
redundant_dict= defaultdict(lambda: defaultdict(int))
for line in open("redundancy_list_without_partially_redundant.txt","r"):
	tokens=line.split()
	redundant_dict[tokens[0]][tokens[1]]=1
novel_list=[]
for topic in topics:
	if topic in topics_allowed:
		for i in range(len(sorted_doc_topic_dict[topic])):
			if redundant_dict[topic][sorted_doc_topic_dict[topic][i]]!=1:
				if i>0:
					# take at most 5 latest docs in case of novel
					novel_list.append(" ".join([topic,sorted_doc_topic_dict[topic][i]]+sorted_doc_topic_dict[topic][max(0,i-5):i]))
with open(write_file,"w") as f:
	f.write("\n".join(novel_list))
