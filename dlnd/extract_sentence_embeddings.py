import torch
import nltk
import os
from nltk.tokenize import sent_tokenize
import numpy as np
import pickle
import time
import sys

sys.path.append("../infersent/")

def cumulative(arr):
	for i in range(len(arr)-1):
		arr[i+1]+=arr[i]
	return arr

if __name__=="__main__":
	dlnd_path = "DLND_08.07.2017(199)/"
	all_direc = [dlnd_path+direc+"/"+direc1+"/" for direc in os.listdir(dlnd_path) if os.path.isdir(dlnd_path+direc) for direc1 in os.listdir(dlnd_path+direc)]
	source_files = [[direc+"source/"+file for file in os.listdir(direc+"source/") if file.endswith(".txt")] for direc in all_direc]
	target_files = [[direc+"target/"+file for file in os.listdir(direc+"target/") if file.endswith(".txt")] for direc in all_direc]
	source_docs = [[sent_tokenize(open(file,"r").read().decode("ascii","ignore")) for file in direc] for direc in source_files]
	target_docs = [[sent_tokenize(open(file,"r").read().decode("ascii","ignore")) for file in direc] for direc in target_files]
	all_sentences = [sent for direc in source_docs for doc in direc for sent in doc]+[sent for direc in target_docs for doc in direc for sent in doc]
	infersent = torch.load("../infersent/encoder/model_1024_attn.pickle")
	infersent.set_glove_path("../../glove.840B.300d.txt")
	print("Infersent started!!")
	start=time.time()
	infersent.build_vocab(all_sentences,tokenize=True)
	all_sentence_vectors = infersent.encode(all_sentences,tokenize=True)
	print("Infersent done!!")
	print("Time taken: "+str(time.time()-start))
	all_sentence_vectors = np.split(all_sentence_vectors,cumulative([sum([len(doc) for doc in direc]) for direc in source_docs]+[sum([len(doc) for doc in direc]) for direc in target_docs]))
	source_sentence_vectors = all_sentence_vectors[:len(source_docs)]
	target_sentence_vectors = all_sentence_vectors[len(source_docs):len(source_docs)+len(target_docs)]
	source_docs_vectors = [np.split(source_sentence_vectors[i],cumulative([len(doc) for doc in source_docs[i]])) for i in range(len(source_sentence_vectors))]
	target_docs_vectors = [np.split(target_sentence_vectors[i],cumulative([len(doc) for doc in target_docs[i]])) for i in range(len(target_sentence_vectors))]
	pickle.dump([source_docs_vectors,target_docs_vectors,source_docs,target_docs,source_files,target_files],open("dlnd_cnn_sentence_embeddings_1024_attn.pickle","wb"))