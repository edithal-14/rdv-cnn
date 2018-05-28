import pickle
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

def build_rdv(t,s):
	return np.concatenate([t,s,np.subtract(t,s),np.multiply(t,s)],axis=0)

def rdv(tm,sm,label):
	match = np.argmin(cdist(tm,sm,metric="cosine"),axis=1)
	vec = np.stack((build_rdv(tm[i],sm[match[i]]) for i in range(len(tm))))
	return [vec,label]

emb = pickle.load(open("apwsj_sentence_embeddings_512.p","rb"))
is_key=defaultdict(int)
for key in emb:
	is_key[key]=1
topics_allowed="q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
topics_allowed=topics_allowed.split(", ")
non_novel = list()
for line in open("redundancy_list_without_partially_redundant.txt","r"):
	tokens = line.split()
	if tokens[0] in topics_allowed:
		non_novel.append([tokens[0]+"/"+tokens[i] for i in range(1,len(tokens))])
novel = list()
for line in open("novel_list_without_partially_redundant.txt","r"):
	tokens = line.split()
	if tokens[0] in topics_allowed:
		novel.append([tokens[0]+"/"+tokens[i] for i in range(1,len(tokens))])
#non_novel = 0 , novel=1
rel_doc_vecs = list()
rdv_files = list()
for case in non_novel:
	# if a target document has more than 200 sentences, we ignore it
	if emb[case[0]].shape[0]>200:
		continue
	file = list()
	if is_key[case[0]]==1:
		file.append(case[0])
		sources = []
		for d in case[1:]:
			if is_key[d]==1:
				sources.append(emb[d])
				file.append(d)
		if len(sources)==0:
			continue
		sources = tuple(sources)
		rel_doc_vecs.append(rdv(emb[case[0]],np.vstack(sources),0))
		file.append("0")
		rdv_files.append(file)
for case in novel:
	# if a target document has more than 200 sentences, we ignore it
	if emb[case[0]].shape[0]>200:
		continue
	file = list()
	if is_key[case[0]]==1:
		file.append(case[0])
		sources = []
		for d in case[1:]:
			if is_key[d]==1:
				sources.append(emb[d])
				file.append(d)
		if len(sources)==0:
			continue
		sources = tuple(sources)
		rel_doc_vecs.append(rdv(emb[case[0]],np.vstack(sources),1))
		file.append("1")
		rdv_files.append(file)
pickle.dump([rel_doc_vecs,rdv_files],open("rdvs_512_without_partially_redundant.pickle","wb"),2) 
