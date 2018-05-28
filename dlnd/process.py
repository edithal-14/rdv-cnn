import pickle
import numpy as np
import collections

if __name__=="__main__":
	cases = pickle.load(open("rdvs.pickle","rb"))
	cv=[1 if np.random.rand() > 0.2 else 0 for i in range(len(cases))]
	labels=[1 if case[1]=="Novel" else 0 for case in cases]
	mapping = collections.defaultdict(int)
	k=1
	for case in cases:
		for i in range(case[0].shape[0]):
			if mapping[str(case[0][i].tolist())]==0:
				mapping[str(case[0][i].tolist())]=k
				k+=1
	for key in mapping.keys():
		mapping[key]-=1
	sentence_index=[[mapping[str(case[0][i].tolist())] for i in range(case[0].shape[0])] for case in cases]
	reverse_mapping=dict([[mapping[key],key] for key in mapping.keys()])
	sentence_vecs=[list(map(float,reverse_mapping[i][1:-1].split(","))) for i in range(k-1)]
	pickle.dump([sentence_index,sentence_vecs,labels,cv], open("mr_dlnd.p", "wb"))