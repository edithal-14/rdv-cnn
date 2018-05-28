import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
from nltk import sent_tokenize

model = Doc2Vec.load("../enwiki_dbow/doc2vec.bin")
data = pickle.load(open("webis_data.pickle","rb"))
new_data=[]
n_cases = len(data)
for i in range(n_cases):
	target_tokens = [token for sent in data[i][0] for token in word_tokenize(sent)]
	target_vec = model.infer_vector(doc_words=target_tokens, alpha=0.1, min_alpha=0.0001, steps=5)
	source_tokens = [token for sent in data[i][1] for token in word_tokenize(sent)]
	source_vec = model.infer_vector(doc_words=source_tokens, alpha=0.1, min_alpha=0.0001, steps=5)
	result_vec = np.concatenate((target_vec,source_vec),axis=0)
	new_data.append([result_vec,data[i][2]])

with open("webis_pv.arff","w") as f:
	f.write("@RELATION webis\n\n")
	for i in range(1,601):
		f.write("@ATTRIBUTE "+str(i)+" numeric\n")
	f.write("@ATTRIBUTE class {1,0}\n\n@DATA\n")
	for row in new_data:
		vec = row[0].tolist()
		f.write(",".join([str(val) for val in vec])+","+str(row[1])+"\n")