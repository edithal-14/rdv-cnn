from collections import defaultdict
lines = open("NoveltyData/apwsj.qrels","r").read().splitlines()
map = defaultdict(list)
for line in lines:
	tokens = line.split()
	if tokens[-1]=="1":
		map[tokens[0]].append(tokens[2])
with open("topic_doc_mapping.txt","w") as f:
	for topic in map:
		f.write(topic+" "+" ".join(map[topic])+"\n")
