from collections import defaultdict
def crawl(red_dict,doc,crawled):
	ans=[]
	for cdoc in red_dict[doc]:
		ans.append(cdoc)
		if crawled[cdoc]==0:
			try:
				red_dict[cdoc]=crawl(red_dict,cdoc,crawled)
				crawled[cdoc]=1
				ans+=red_dict[cdoc]
			except:
				crawled[cdoc]=1
	return ans

wf = "redundancy_list_without_partially_redundant.txt"
topics_allowed="q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
topics_allowed=topics_allowed.split(", ")
red_dict = dict()
allow_partially_redundant = 0
for line in open("NoveltyData/redundancy.apwsj.result","r"):
	tokens = line.split()
	if tokens[2]=="?":
		if allow_partially_redundant==1:
			red_dict[tokens[0]+"/"+tokens[1]]=[tokens[0]+"/"+i for i in tokens[3:]]
	else:
		red_dict[tokens[0]+"/"+tokens[1]]=[tokens[0]+"/"+i for i in tokens[2:]]
crawled=defaultdict(int)
for doc in red_dict:
	if crawled[doc]==0:
		red_dict[doc]=crawl(red_dict,doc,crawled)
		crawled[doc]=1
with open(wf,"w") as f:
	for doc in red_dict:
		if doc.split("/")[0] in topics_allowed:
			f.write(" ".join(doc.split("/")+[i.split("/")[1] for i in red_dict[doc]])+"\n")
