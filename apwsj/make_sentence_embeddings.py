import torch
import nltk
import os
import numpy as np
import pickle
import time
from nltk.tokenize import sent_tokenize

def cumulative(arr):
	for i in range(len(arr)-1):
		arr[i+1]+=arr[i]
	return arr

if __name__=="__main__":
	doc_dir="../apwsj/apwsj_parsed_documents/"
	topics_allowed="q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
	topics_allowed=topics_allowed.split(", ")
	topics = [top for top in os.listdir(doc_dir) if top in topics_allowed]
	file_names = [t+"/"+f for t in topics for f in os.listdir(doc_dir+t+"/")]
	files = [sent_tokenize(open(doc_dir+f,"r").read().encode("ascii","ignore").decode("ascii")) for f in file_names]
	all_sentences = [sent for file in files for sent in file]
	infersent = torch.load("encoder/model_1024.pickle")
	infersent.set_glove_path("../../glove.840B.300d.txt")
	print("Infersent started!!")
	start=time.time()
	infersent.build_vocab(all_sentences,tokenize=True)
	all_sentence_vectors = infersent.encode(all_sentences,tokenize=True)
	print("Infersent done!!")
	print("Time taken: "+str(time.time()-start))
	all_sentence_vectors=np.split(all_sentence_vectors,cumulative([len(file) for file in files]))
	sentence_embeddings = dict()
	for i in range(len(file_names)):
		sentence_embeddings[file_names[i]]=all_sentence_vectors[i]
	pickle.dump(sentence_embeddings,open("apwsj_sentence_embeddings.p","wb"),2)
