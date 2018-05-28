import pickle
import numpy as np
import collections
from random import shuffle
import numpy as np

if __name__=="__main__":
	cases,rdv_files = pickle.load(open("rdvs_512_without_partially_redundant.pickle","rb"))
	labels=[case[1] for case in cases]
	cv=[None]*len(cases)
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
	cv_dict=dict()
	for i in range(len(labels)):
		try:
				cv_dict[cv[i]].append(labels[i])
		except:
				cv_dict[cv[i]]=[labels[i]]
	print([i for i in cv_dict])
	print([len(cv_dict[key]) for key in cv_dict.keys()])
	print([sum(cv_dict[key]) for key in cv_dict.keys()])
	all_rows = np.vstack((case[0] for case in cases))
	unique_rows = list({tuple(row) for row in all_rows})
	mapping = {unique_rows[i]:i for i in range(len(unique_rows))}
	sentence_index = [[mapping[tuple(row)] for row in case[0]] for case in cases]
	sentence_vecs = np.vstack((row for row in unique_rows))
	pickle.dump([sentence_index,sentence_vecs,labels,cv], open("mr_apwsj_512_without_partially_redundant.p", "wb"))
