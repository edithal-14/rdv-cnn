import torch
import nltk
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import time
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append("../infersent/")

def cumulative(arr):
	for i in range(len(arr)-1):
		arr[i+1]+=arr[i]
	return arr

if __name__=="__main__":
	data = pickle.load(open("webis_data.pickle","rb"))
	sources = [[sent for sent in case[0]] for case in data]
	targets = [[sent for sent in case[1]] for case in data]
	gold = [case[2] for case in data]
	print("data loaded...")

	source_sentences = [sent.decode("ascii","ignore") for case in data for sent in case[0]]
	print("loading infersent...")
	infersent = torch.load("../infersent/encoder/model_2048_attn.pickle")
	infersent.set_glove_path("../../glove.840B.300d.txt")
	print("Infersent started!!")
	start=time.time()
	infersent.build_vocab(source_sentences,tokenize=True)
	source_sentence_vectors = infersent.encode(source_sentences,tokenize=True)
	print("Infersent done!!")
	print("Time taken: "+str(time.time()-start))
	source_vecs = np.split(source_sentence_vectors,cumulative([len(case[0]) for case in data])[:-1])

	target_sentences = [sent.decode("ascii","ignore") for case in data for sent in case[1]]
	print("loading infersent...")
	infersent = torch.load("../infersent/encoder/model_2048_attn.pickle")
	infersent.set_glove_path("../../glove.840B.300d.txt")
	print("Infersent started!!")
	start=time.time()
	infersent.build_vocab(target_sentences,tokenize=True)
	target_sentence_vectors = infersent.encode(target_sentences,tokenize=True)
	print("Infersent done!!")
	print("Time taken: "+str(time.time()-start))
	target_vecs = np.split(target_sentence_vectors,cumulative([len(case[1]) for case in data])[:-1])

	pickle.dump([targets,sources,target_vecs,source_vecs,gold],open("webis_embeddings_data_2048_attn.p","wb"))
