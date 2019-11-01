import torch
import nltk
import os
from nltk.tokenize import sent_tokenize
import numpy as np
import pickle
import time
import sys
import xml.etree.ElementTree as ET

sys.path.append("../sentence_encoder/")

def cumulative(arr):
	for i in range(len(arr)-1):
		arr[i+1]+=arr[i]
	return arr

if __name__=="__main__":
	dlnd_path = "DLND_08.07.2017(199)/"
	all_direc = [dlnd_path+direc+"/"+direc1+"/" for direc in os.listdir(dlnd_path) if os.path.isdir(dlnd_path+direc) for direc1 in os.listdir(dlnd_path+direc)]
	source_files = [[direc+"source/"+file for file in os.listdir(direc+"source/") if file.endswith(".txt")] for direc in all_direc]
	target_files = [[direc+"target/"+file for file in os.listdir(direc+"target/") if file.endswith(".txt")] for direc in all_direc]
	sources = []
	targets = []
	gold = [[tag.attrib["DLA"] for tag in ET.parse(target_files[i][j][:-4]+".xml").findall("feature") if "DLA" in tag.attrib.keys()][0] for i in range(len(target_files)) for j in range(len(target_files[i]))]
	gold = [1 if label=="Novel" else 0 for label in gold]
	for dir_n in range(len(target_files)):
		sdata = [sent for doc in [sent_tokenize(open(sfile,"r").read().decode("ascii","ignore")) for sfile in source_files[dir_n]] for sent in doc]
		for tfile in target_files[dir_n]:
			targets.append(sent_tokenize(open(tfile,"r").read().decode("ascii","ignore")))
			sources.append(sdata)

	source_sentences = [sent for doc in sources for sent in doc]
	print("loading infersent...")
	sentence_encoder = torch.load("../sentence_encoder/encoder/model_2048_attn.pickle")
	infersent.set_glove_path("../../glove.840B.300d.txt")
	print("Infersent started!!")
	start=time.time()
	infersent.build_vocab(source_sentences,tokenize=True)
	source_sentence_vectors = infersent.encode(source_sentences,tokenize=True)
	print("Infersent done!!")
	print("Time taken: "+str(time.time()-start))
	source_vecs = np.split(source_sentence_vectors,cumulative([len(doc) for doc in sources])[:-1])

	target_sentences = [sent for doc in targets for sent in doc]
	print("loading infersent...")
	sentence_encoder = torch.load("../sentence_encoder/encoder/model_2048_attn.pickle")
	infersent.set_glove_path("../../glove.840B.300d.txt")
	print("Infersent started!!")
	start=time.time()
	infersent.build_vocab(target_sentences,tokenize=True)
	target_sentence_vectors = infersent.encode(target_sentences,tokenize=True)
	print("Infersent done!!")
	print("Time taken: "+str(time.time()-start))
	target_vecs = np.split(target_sentence_vectors,cumulative([len(doc) for doc in targets])[:-1])

	pickle.dump([targets,sources,target_vecs,source_vecs,gold],open("dlnd_data_2048_attn.p","wb"))
