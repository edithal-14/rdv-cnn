from nltk import ngrams
import os
import numpy as np
import pickle
import time
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.tag.stanford import NERTagger
from gensim.models.doc2vec import Doc2Vec
from gensim.summarization import summarize
from gensim.summarization import keywords
from gensim.models import KeyedVectors
from pyemd import emd
import string
from scipy.spatial.distance import cosine
import math
# hand crafted features
#1. Paragraph vector cosine
#2. Lexical similarity of upto 5 grams
#3. Ner similarity
#4. Rake keyword similarity
#5. New word count
#6. Word2vec concatenation similarity
#7. concept similarity using textrank
#8. kl divergence
#9. MMI

model = Doc2Vec.load("../enwiki_dbow/doc2vec.bin")
stoplist = set(stopwords.words('english') + list(string.punctuation))
ner = NERTagger("../stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz","../stanford-ner-2017-06-09/stanford-ner.jar")
w2v_model = KeyedVectors.load_word2vec_format('../w2v.bin', binary=True)

def ngrams_similarity(target_text,source_text):
	global stoplist
	target_tokens = [token.lower() for token in word_tokenize(target_text) if token.lower() not in stoplist]
	source_tokens = [token.lower() for token in word_tokenize(source_text) if token.lower() not in stoplist]
	target_ngrams = list(ngrams(target_tokens,2))+list(ngrams(target_tokens,3))+list(ngrams(target_tokens,8))
	source_ngrams = list(ngrams(source_tokens,2))+list(ngrams(target_tokens,3))+list(ngrams(target_tokens,8))
	similarity = len([gram for gram in target_ngrams if gram in source_ngrams])/float(len([gram for gram in target_ngrams]))
	return similarity

def doc2vec_cosine(target_text,source_text_list):
	global model,stoplist
	target_tokens = [token.lower() for token in word_tokenize(target_text) if token.lower() not in stoplist]
	target_vector = model.infer_vector(doc_words=target_tokens, alpha=0.1, min_alpha=0.0001, steps=5)
	min_cosine_distance = float('inf')
	for source_text in source_text_list:
		source_tokens = [token.lower() for token in word_tokenize(source_text) if token.lower() not in stoplist]
		source_vector = model.infer_vector(doc_words=source_tokens, alpha=0.1, min_alpha=0.0001, steps=5)
		cosine_distance = cosine(target_vector,source_vector)
		if cosine_distance < min_cosine_distance:
			min_cosine_distance = cosine_distance
	return min_cosine_distance

def ner_score(target_text,source_text):
	global ner
	target_tokens = [token for token in word_tokenize(target_text)]
	source_tokens = [token for token in word_tokenize(source_text)]
	target_tagged = [i for i in ner.tag(target_tokens) if i[1]!='O']
	source_tagged = [i for i in ner.tag(source_tokens) if i[1]!='O']
	if len(target_tagged)==0 or len(source_tagged)==0:
		return 0
	else:
		ner_score_val = len([i for i in target_tagged if i in source_tagged])/float(len(target_tagged))
	return ner_score_val

def nwc_score(target_text,source_text):
	target_tokens = [token.lower() for token in word_tokenize(target_text) if token.lower() not in stoplist]
	source_tokens = [token.lower() for token in word_tokenize(source_text) if token.lower() not in stoplist]
	score = len([i for i in target_tokens if i not in source_tokens])/float(len(target_tokens))

def w2v_score(target_text,source_text_list):
	global w2v_model
	target_tokens = [token.lower() for token in word_tokenize(target_text) if token.lower() not in stoplist]
	target_vecs=[]
	for token in target_tokens:
		try:
			target_vecs.append(np.array(w2v_model[token]))
		except:
			continue
	if len(target_vecs)==0:
		return 0
	else:
		target_vector = np.concatenate(target_vecs,axis=0)
	min_cosine_distance = float('inf')
	for source_text in source_text_list:
		source_tokens = [token.lower() for token in word_tokenize(source_text) if token.lower() not in stoplist]
		source_vecs=[]
		for token in source_tokens:
			try:
				source_vecs.append(np.array(w2v_model[token]))
			except:
				continue
		if len(source_vecs)==0:
			return 0
		else:
			source_vector = np.concatenate(source_vecs,axis=0)
		min_length = min(len(target_vector),len(source_vector))
		cosine_distance = cosine(target_vector[:min_length],source_vector[:min_length])
		if cosine_distance < min_cosine_distance:
			min_cosine_distance = cosine_distance
	return min_cosine_distance

def keyword_score(target_keys,source_keys):
	target_keys = target_keys.split()
	source_keys = source_keys.split()
	return len([key for key in target_keys if key not in source_keys])/float(len(target_keys))

def kl_score(target_text,source_text):
	target_tokens = [token.lower() for token in word_tokenize(target_text) if token.lower() not in stoplist]
	source_tokens = [token.lower() for token in word_tokenize(source_text) if token.lower() not in stoplist]
	target_dict = dict([[token,len([i for i in target_tokens if i==token])] for token in set(target_tokens)])
	source_dict = dict([[token,len([i for i in source_tokens if i==token])] for token in set(source_tokens)])
	if len(target_dict)==0 or len(source_dict)==0:
		return 1e33
	ssum = float(sum(source_dict.values()))
	tsum = float(sum(target_dict.values()))
	diff = len(set(source_dict.keys()).difference(set(target_dict.keys())))
	epsilon = 0.001* min(min(source_dict.values())/ssum,min(target_dict.values())/tsum)
	gamma = 1 - diff*epsilon
	vocab = set(source_dict.keys()).union(set(target_dict.keys()))
	div=0
	for term in vocab:
		try:
			pts = source_dict[term]/ssum
		except:
			pts = epsilon
		try:
			ptt = gamma * (target_dict[term]/tsum)
		except:
			ptt = epsilon
		div+=(pts-ptt)*math.log(pts/ptt)
	return div

def mmi_score(target_text,source_text):
	mmi_dict = dict()
	target_lines = sent_tokenize(target_text)
	source_lines = sent_tokenize(source_text)
	for idxs,sents in enumerate(source_lines):
		for idxt,sentt in enumerate(target_lines):
			source_tokens = word_tokenize(sents)
			target_tokens = word_tokenize(sentt)
			wmd = w2v_model.wmdistance(source_tokens,target_tokens)
			try:
				mmi_dict[idxs].append([idxt,sentt,wmd])
			except:
				mmi_dict[idxs] = [[idxt,sentt,wmd]]
	mp = len(Median_partition(mmi_dict))/float(len(target_lines))
	lp = len(Largest_partition(mmi_dict))/float(len(target_lines))
	ap = len(All_partition(mmi_dict))/float(len(target_lines))
	return [mp,lp,ap]

def Median_partition(mmi_dict):
	max_divergent = []
	max_similar = []
	max_similar_set = set()	
	max_divergent_set = set()
	for keys in mmi_dict.keys(): 
		key_len = len(mmi_dict[keys])
		mmi_dict[keys] = sorted(mmi_dict[keys],key=itemgetter(2))

		max_similar.append(mmi_dict[keys][0])
		for sen_tup in mmi_dict[keys][slice(int(0.5*key_len),int(key_len))]:
			max_divergent.append(sen_tup)
	for sen_tuple in max_similar:
		max_similar_set.add(sen_tuple[0])
	for sen_tuple in max_divergent:
		max_divergent_set.add(sen_tuple[0])	
	novel_set = max_divergent_set - max_similar_set
	return novel_set

def Largest_partition(mmi_dict):
	max_divergent = []
	max_similar = []
	max_similar_set = set()	
	max_divergent_set = set()
	for keys in mmi_dict.keys():
		max_diff = 0
		index = 0  
		key_len = len(mmi_dict[keys])
		mmi_dict[keys] = sorted(mmi_dict[keys],key=itemgetter(2))
		for i in range(len(mmi_dict[keys])-1):
			if ( (mmi_dict[keys][i+1][2]-mmi_dict[keys][i][2]) > max_diff ) :
				max_diff = (mmi_dict[keys][i+1][2]-mmi_dict[keys][i][2])
				index = i

		max_similar.append(mmi_dict[keys][0])
		for sen_tup in mmi_dict[keys][slice(index,int(key_len))]:
				max_divergent.append(sen_tup)
	for sen_tuple in max_similar:
		max_similar_set.add(sen_tuple[0])
	for sen_tuple in max_divergent:
		max_divergent_set.add(sen_tuple[0])
	novel_set = max_divergent_set - max_similar_set
	return novel_set	

def All_partition(mmi_dict):
	max_divergent = []
	max_similar = []
	max_similar_set = set()	
	max_divergent_set = set()
	for keys in mmi_dict.keys(): 
		key_len = len(mmi_dict[keys])
		mmi_dict[keys] = sorted(mmi_dict[keys],key=itemgetter(2))

		max_similar.append(mmi_dict[keys][0])
		for sen_tup in mmi_dict[keys][1:]:
			max_divergent.append(sen_tup)
	for sen_tuple in max_similar:
		max_similar_set.add(sen_tuple[0])
	for sen_tuple in max_divergent:
		max_divergent_set.add(sen_tuple[0])
	novel_set = max_divergent_set - max_similar_set	
	return novel_set


dlnd_path = "../DLND_08.07.2017(199)/"
all_direc = [dlnd_path+direc+"/"+direc1+"/" for direc in os.listdir(dlnd_path) if os.path.isdir(dlnd_path+direc) for direc1 in os.listdir(dlnd_path+direc)]
source_files = [[direc+"source/"+file for file in os.listdir(direc+"source/") if file.endswith(".txt")] for direc in all_direc]
target_files = [[direc+"target/"+file for file in os.listdir(direc+"target/") if file.endswith(".txt")] for direc in all_direc]
source_docs = [[sent_tokenize(open(file,"r").read().decode("ascii","ignore")) for file in direc] for direc in source_files]
target_docs = [[sent_tokenize(open(file,"r").read().decode("ascii","ignore")) for file in direc] for direc in target_files]
lexical_similarity = [ngrams_similarity(open(target_files[i][j],"r").read()," . ".join([open(source_files[i][j],"r").read() for j in range(len(source_files[i]))])) for i in range(len(target_files)) for j in range(len(target_files[i]))]
pv_cosine = [doc2vec_cosine(open(target_files[i][j],"r").read(),[open(source_files[i][j],"r").read() for j in range(len(source_files[i]))]) for i in range(len(target_files)) for j in range(len(target_files[i]))]
ner_similarity = [ner_score(open(target_files[i][j],"r").read()," . ".join([open(source_files[i][j],"r").read() for j in range(len(source_files[i]))])) for i in range(len(target_files)) for j in range(len(target_files[i]))]
nwc = [nwc_score(open(target_files[i][j],"r").read()," . ".join([open(source_files[i][j],"r").read() for j in range(len(source_files[i]))])) for i in range(len(target_files)) for j in range(len(target_files[i]))]
w2v_cosine = [w2v_score(open(target_files[i][j],"r").read(),[open(source_files[i][j],"r").read() for j in range(len(source_files[i]))]) for i in range(len(target_files)) for j in range(len(target_files[i]))]
keyword_sim = [keyword_score(keywords(open(target_files[i][j],"r").read()),keywords(" . ".join([(open(source_files[i][j],"r").read()) for j in range(len(source_files[i]))]))) for i in range(len(target_files)) for j in range(len(target_files[i]))]
summary_cosine=[]
for i in range(len(target_files)):
	for j in range(len(target_files[i])):
		target_text = open(target_files[i][j],"r").read()
		source_text_list = [open(source_files[i][j],"r").read() for j in range(len(source_files[i]))]
		if len(sent_tokenize(target_text))>1:
			target_text_sum = summarize(target_text)
		else:
			target_text_sum = target_text
		source_text_list_sum = []
		for ss in source_text_list:
			if len(sent_tokenize(ss)) > 1:
				source_text_list_sum.append(summarize(ss))
			else:
				source_text_list_sum.append(ss)
		summary_cosine.append(w2v_cosine(target_text_sum,source_text_list_sum))
kld = [kl_score(open(target_files[i][j],"r").read()," . ".join([open(source_files[i][j],"r").read() for j in range(len(source_files[i]))])) for i in range(len(target_files)) for j in range(len(target_files[i]))]
mmi = [mmi_score(open(target_files[i][j],"r").read()," . ".join([open(source_files[i][j],"r").read() for j in range(len(source_files[i]))])) for i in range(len(target_files)) for j in range(len(target_files[i])) ]
pickle.dump([source_files,target_files,source_docs,target_docs,lexical_similarity,pv_cosine,ner_similarity,nwc,w2v_cosine,summary_cosine,keyword_sim,kld,mmi],open("dlnd_hcf.pickle","wb"))