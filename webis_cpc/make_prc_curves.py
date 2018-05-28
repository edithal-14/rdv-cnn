import pickle
from sklearn.metrics import precision_recall_curve
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
data_file = "webis_prc_curves_data.p"

def train_lr(features,labels):
	features = np.array(features)
	class_order = np.unique(labels)
	print("Class order: "+str(class_order))
	lr = LogisticRegression(verbose=2,n_jobs=-1)
	predictions = []
	probs = []
	gold = []
	kfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 9274)
	for train,test in kfold.split(np.zeros(len(labels)),labels):
		lr.fit(features[train],np.array(labels)[train].tolist())
		prob = lr.predict_proba(features[test])
		preds = [class_order[i] for i in np.argmax(prob,axis=1)]
		prob = prob.tolist()
		predictions.append(preds)
		probs.append(prob)
		gold.append(np.array(labels)[test].tolist())
	probs = [j for i in probs for j in i]
	gold = np.array([j for i in gold for j in i])
	predictions = np.array([j for i in predictions for j in i])
	cf = confusion_matrix(gold,predictions,labels=[0,1])
	acc= accuracy_score(gold,predictions)
	p,r,f,_=precision_recall_fscore_support(gold,predictions,labels=[0,1])
	print("\nConfusion matrix:\n"+str(cf))
	print("\nAccuracy: "+str(acc))
	print("\nClass wise precisions: "+str(p))
	print("Class wise recalls: "+str(r))
	print("Class wise fscores: "+str(f))
	return probs

if not os.path.exists(data_file):
	preds = pickle.load(open("webis_1024_cnn_output.pickle","rb"))
	# predictions = [j for i in preds for j in i[1]]
	golds = [j for i in preds for j in i[2]]
	probs_cnn = [j for i in preds for j in i[0]]
	del preds
	probs_cnn = train_lr(probs_cnn,golds)

	_,probs_set_diff,probs_geo_diff,probs_tfidf_novelty_score,probs_kl_div,probs_pv,_ = pickle.load(open("webis_baselines_class_probs.p","rb"))

	_,_,class_probs,_,_,_ = pickle.load(open("webis_ten_fold_progress_bilstm_mlp_baseline.p","rb"))
	probs_bilstm_mlp = [j for i in class_probs for j in i]

	pickle.dump([probs_set_diff,probs_geo_diff,probs_tfidf_novelty_score,probs_kl_div,probs_pv,probs_bilstm_mlp,probs_cnn,golds],open(data_file,"wb"))
else:
	probs_set_diff,probs_geo_diff,probs_tfidf_novelty_score,probs_kl_div,probs_pv,probs_bilstm_mlp,probs_cnn,golds = pickle.load(open(data_file,"rb"))

p_set_diff,r_set_diff,_ = precision_recall_curve(golds,[i[1] for i in probs_set_diff])
p_geo_diff,r_geo_diff,_ = precision_recall_curve(golds,[i[1] for i in probs_geo_diff])
p_tfidf_novelty_score,r_tfidf_novelty_score,_ = precision_recall_curve(golds,[i[1] for i in probs_tfidf_novelty_score])
p_kl_div,r_kl_div,_ = precision_recall_curve(golds,[i[1] for i in probs_kl_div])
p_pv,r_pv,_ = precision_recall_curve(golds,[i[1] for i in probs_pv])
p_bilstm_mlp,r_bilstm_mlp,_ = precision_recall_curve(golds,[i[1] for i in probs_bilstm_mlp])
p_cnn,r_cnn,_ = precision_recall_curve(golds,[i[1] for i in probs_cnn])

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Novel class Precision-Recall curve')
plt_set_diff = plt.plot(r_set_diff,p_set_diff,'-', label="Set Difference")
plt_geo_diff = plt.plot(r_geo_diff,p_geo_diff,'-', label="Geometric Difference")
plt_tfidf_novelty_score = plt.plot(r_tfidf_novelty_score,p_tfidf_novelty_score,'-', label="IDF novelty score")
plt_kl_div = plt.plot(r_kl_div,p_kl_div,'-', label="KL divergence ( Lang. model )")
plt_pv = plt.plot(r_pv,p_pv,'-', label="Paragraph vector")
plt_bilstm_mlp = plt.plot(r_bilstm_mlp,p_bilstm_mlp,'-', label="BiLSTM + MLP")
plt_cnn = plt.plot(r_cnn,p_cnn,'-', label="RDV + CNN")
#handles=[plt_set_diff,plt_geo_diff,plt_tfidf_novelty_score,plt_kl_div,plt_pv,plt_bilstm_mlp,plt_cnn]
plt.legend(loc='best')
# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
plt.show()