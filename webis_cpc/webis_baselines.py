import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from gensim.models.doc2vec import Doc2Vec
import sys

model = Doc2Vec.load("../../enwiki_dbow/doc2vec.bin")
stopwords = list(string.punctuation)+list(set(stopwords.words('english')))
tfidf_vec = TfidfVectorizer(decode_error="ignore",lowercase=False,stop_words=stopwords,sublinear_tf=True,smooth_idf=True)
tfidf_vec1 = TfidfVectorizer(decode_error="ignore",lowercase=False,stop_words=stopwords,smooth_idf=True)
count_vec = CountVectorizer(decode_error="ignore",lowercase=False,stop_words=stopwords)

def calc_set_diff(source_sent,target_sent):
	global tfidf_vec
	source_data = " . ".join(source_sent)
	target_data = " . ".join(target_sent)
	doc_term = tfidf_vec.fit_transform([source_data,target_data])
	source_set = set(np.nonzero(doc_term[0])[1].tolist())
	target_set = set(np.nonzero(doc_term[1])[1].tolist())
	diff = len(target_set-source_set)
	return diff

def calc_geo_diff(source_sent,target_sent):
	global tfidf_vec
	source_data = " . ".join(source_sent)
	target_data = " . ".join(target_sent)
	doc_term = tfidf_vec.fit_transform([source_data,target_data])
	cos = float(cosine_similarity(doc_term[0],doc_term[1])[0][0])
	return cos

def calc_kl_div(source_sent,target_sent):
	global count_vec
	source_data = " . ".join(source_sent)
	target_data = " . ".join(target_sent)
	doc_term = count_vec.fit_transform([source_data,target_data])
	source_count = doc_term[0].todense()
	target_count = doc_term[1].todense()
	source_dist = [0.5 if source_count[0,i]==0 else source_count[0,i] for i in range(source_count.shape[1])]
	target_dist = [0.5 if target_count[0,i]==0 else target_count[0,i]  for i in range(target_count.shape[1])]
	kl = entropy(target_dist,source_dist)
	return kl

def calc_tfidf_novelty_score(source_sent,target_sent):
	global tfidf_vec1,count_vec
	source_data = " . ".join(source_sent)
	target_data = " . ".join(target_sent)
	doc_term = tfidf_vec1.fit_transform([source_data,target_data])
	doc_term1 = count_vec.fit_transform([source_data,target_data])
	if doc_term1.sum(axis=1)[1,0]==0:
		score=0.0
	else:
		score = doc_term.sum(axis=1)[1,0]/float(doc_term1.sum(axis=1)[1,0])
	return score

def calc_pv(source_sents,target_sents):
	source_vec = model.infer_vector(doc_words=[token for sent in source_sents for token in word_tokenize(sent.decode('ascii','ignore')) if token not in stopwords], alpha=0.1, min_alpha=0.0001, steps=5)
	target_vec = model.infer_vector(doc_words=[token for sent in target_sents for token in word_tokenize(sent.decode('ascii','ignore')) if token not in stopwords], alpha=0.1, min_alpha=0.0001, steps=5)
	return np.concatenate((target_vec,source_vec),axis=0)

def make_cv_10_fold(labels):
	cv = [None]*n_cases
	pos_rows = []
	neg_rows = []
	for n,l in enumerate(labels):
		if l==1:
			pos_rows.append(n)
		elif l==0:
			neg_rows.append(n)
	shuffle(pos_rows)
	shuffle(neg_rows)
	for i in range(10):
		for n in pos_rows[int((len(pos_rows)*i)/float(10)):int((len(pos_rows)*(i+1))/float(10))]:
			cv[n] = i
	for i in range(10):
		for n in neg_rows[int((len(neg_rows)*i)/float(10)):int((len(neg_rows)*(i+1))/float(10))]:
			cv[n] = i
	##
	cv_dict = dict()
	for i in range(len(cv)):
		try:
			cv_dict[cv[i]].append(labels[i])
		except:
			cv_dict[cv[i]]=[labels[i]]
	print([len(cv_dict[key]) for key in cv_dict.keys()])
	print([sum(cv_dict[key]) for key in cv_dict.keys()])
	##
	return cv

def train_lr(features):
	global labels,cv
	class_order = np.unique(labels)
	lr = LogisticRegression(verbose=2,n_jobs=-1)
	predictions = [None]*len(labels)
	probs = [None]*len(labels)
	for curr_cv in range(10):
		trainX=[]
		trainY=[]
		testX=[]
		testY=[]
		for i in range(len(cv)):
			if cv[i]==curr_cv:
				testX.append(features[i])
				testY.append(labels[i])
			else:
				trainX.append(features[i])
				trainY.append(labels[i])
		trainX = np.array(trainX)
		if len(trainX.shape)==1:
			trainX = trainX.reshape(-1,1)
		testX = np.array(testX)
		if len(testX.shape)==1:
			testX = testX.reshape(-1,1)
		lr.fit(trainX,trainY)
		prob = lr.predict_proba(testX)
		preds = [class_order[i] for i in np.argmax(prob,axis=1)]
		prob = prob.tolist()
		correct_indices = [i for i in range(len(cv)) if cv[i]==curr_cv]
		for i in range(len(correct_indices)):
			predictions[correct_indices[i]] = preds[i]
			probs[correct_indices[i]] = prob[i]
	gold = np.array(labels)
	predictions = np.array(predictions)
	cf = confusion_matrix(gold,predictions,labels=[0,1])
	acc= accuracy_score(gold,predictions)
	p,r,f,_=precision_recall_fscore_support(gold,predictions,labels=[0,1])
	print("\nConfusion matrix:\n"+str(cf))
	print("\nAccuracy: "+str(acc))
	print("\nClass wise precisions: "+str(p))
	print("Class wise recalls: "+str(r))
	print("Class wise fscores: "+str(f))
	return probs

if __name__=="__main__":
	data = pickle.load(open("webis_data.pickle","rb"))
	n_cases = len(data)
	labels = [data[i][2] for i in range(n_cases)]
	class_order = np.unique(labels)
	cv = make_cv_10_fold(labels)
	print("\nSET DIFFERENCE METRIC\n")
	set_diff = [calc_set_diff(data[i][0],data[i][1]) for i in range(n_cases)]
	probs_set_diff = train_lr(set_diff)
	print("\nGEO DIFFERENCE METRIC\n")
	geo_diff = [calc_geo_diff(data[i][0],data[i][1]) for i in range(n_cases)]
	probs_geo_diff = train_lr(geo_diff)
	print("\nTFIDF NOVELTY SCORE METRIC\n")
	tfidf_novelty_score = [calc_tfidf_novelty_score(data[i][0],data[i][1]) for i in range(n_cases)]
	probs_tfidf_novelty_score = train_lr(tfidf_novelty_score)
	print("\nKL DIVERGENCE METRIC\n")
	kl_div = [calc_kl_div(data[i][0],data[i][1]) for i in range(n_cases)]
	probs_kl_div = train_lr(kl_div)
	print("\nPARAGRAPH VECTOR + LR\n")
	pv = [calc_pv(data[i][0],data[i][1]) for i in range(n_cases)]
	probs_pv = train_lr(pv)
	pickle.dump([class_order,probs_set_diff,probs_geo_diff,probs_tfidf_novelty_score,probs_kl_div,probs_pv,labels],open("webis_baselines_class_probs.p","wb"))