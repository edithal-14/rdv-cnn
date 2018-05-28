import pickle
import numpy as np
from scipy.spatial.distance import cdist
from random import shuffle

def opr(s,t):
	return np.concatenate((t,s,np.abs(t-s),np.multiply(t,s)))

def build_rdv(source_mat,target_mat):
	match = np.argmin(cdist(target_mat,source_mat,metric="cosine"),axis=1)
	return np.vstack((opr(source_mat[match[i]],target_mat[i]) for i in range(target_mat.shape[0])))

targets,sources,target_vecs,source_vecs,gold = pickle.load(open("webis_embeddings_data_1024_attn.p","rb"))
labels = gold
data = [[source_vecs[i],target_vecs[i],gold[i]] for i in range(len(gold))]
rdv = [[build_rdv(data[i][0],data[i][1]),data[i][2]] for i in range(len(data))]
mapping = dict()
k = 0
for case in rdv:
	for row in range(case[0].shape[0]):
		key=str(case[0][row].tolist())
		try:
			dummy_var = mapping[key]
		except:
			mapping[key] = k
			k+=1
print(str(k))
sentence_index=[[mapping[str(case[0][i].tolist())] for i in range(case[0].shape[0])] for case in rdv]
reverse_mapping=dict([[mapping[key],key] for key in mapping.keys()])
sentence_vecs=[list(map(float,reverse_mapping[i][1:-1].split(","))) for i in range(k)]
cv=[None]*(len(rdv))
pos_class=[]
neg_class=[]
for n,l in [li for li in enumerate(labels)]:
	if l==1:
		pos_class.append(n)
	elif l==0:
		neg_class.append(n)
shuffle(pos_class)
shuffle(neg_class)
for i in range(10):
	for n in pos_class[int((len(pos_class)*i)/10):int((len(pos_class)*(i+1))/10)]:
		cv[n]=i
for i in range(10):
	for n in neg_class[int((len(neg_class)*i)/10):int((len(neg_class)*(i+1))/10)]:
		cv[n]=i
labels=[case[1] for case in rdv]
pickle.dump([sentence_index,sentence_vecs,labels,cv], open("mr_webis_1024_attn.p", "wb"))
