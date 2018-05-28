import torch
torch.cuda.set_device(0)
import pickle
import os
import nltk
import xml.etree.ElementTree as ET
import numpy as np
import sys
sys.path.append("../infersent/")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

targets = list()
sources = list()
gold = list()
topics_allowed="q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
topics_allowed=topics_allowed.split(", ")
for line in open("redundancy_list.txt","r"):
	tokens = line.split()
	if tokens[0] in topics_allowed:
		targets.append(nltk.sent_tokenize(open("apwsj_parsed_documents/"+tokens[0]+"/"+tokens[1],"r").read().decode("utf-8","ignore")))
		sources.append(nltk.sent_tokenize(" . ".join([open("apwsj_parsed_documents/"+tokens[0]+"/"+tokens[i],"r").read().decode("utf-8","ignore") for i in range(2,len(tokens))])))
		# 1 for novel 0 for non-novel
		gold.append(0)
for line in open("novel_list.txt","r"):
	tokens = line.split()
	if tokens[0] in topics_allowed:
		targets.append(nltk.sent_tokenize(open("apwsj_parsed_documents/"+tokens[0]+"/"+tokens[1],"r").read().decode("utf-8","ignore")))
		sources.append(nltk.sent_tokenize(" . ".join([open("apwsj_parsed_documents/"+tokens[0]+"/"+tokens[i],"r").read().decode("utf-8","ignore") for i in range(2,len(tokens))])))
		# 1 for novel 0 for non-novel
		gold.append(1)

target_sentences = [sent for doc in targets for sent in doc]
infersent = torch.load("../infersent/encoder/model_2048_attn.pickle")
infersent.set_glove_path("../../glove.840B.300d.txt")
infersent.build_vocab(target_sentences,tokenize=True)
all_vecs = infersent.encode(target_sentences,tokenize=True)
target_vecs = []
i=0
for doc in targets:
	target_vecs.append(np.array(all_vecs[i:i+len(doc)]))
	i+=len(doc)

source_sentences = [sent for doc in sources for sent in doc]
infersent = torch.load("../infersent/encoder/model_2048_attn.pickle")
infersent.set_glove_path("../../glove.840B.300d.txt")
infersent.build_vocab(source_sentences,tokenize=True)
all_vecs = infersent.encode(source_sentences,tokenize=True)
source_vecs = []
i = 0
for doc in sources:
	source_vecs.append(np.array(all_vecs[i:i+len(doc)]))
	i+=len(doc)

pickle.dump([targets,sources,target_vecs,source_vecs,gold],open("apwsj_data_inner_attention_snli.p","wb"))