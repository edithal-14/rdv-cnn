import pickle
from sklearn.metrics import precision_recall_curve
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

data_file = "dlnd_prc_curves_data.p"

if not os.path.exists(data_file):
	preds = pickle.load(open("dlnd_1024_cnn_output.pickle","rb"))
	# predictions = [j for i in preds for j in i[1]]
	golds = [j for i in preds for j in i[2]]
	probs_cnn = [j for i in preds for j in i[0]]
	del preds

	_,probs_set_diff,probs_geo_diff,probs_tfidf_novelty_score,probs_kl_div,probs_pv,_ = pickle.load(open("dlnd_baselines_class_probs.p","rb"))

	_,_,class_probs,_,_,_ = pickle.load(open("dlnd_ten_fold_progress_bilstm_mlp_baseline.p","rb"))
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
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()
