import pickle
from sklearn.metrics import precision_recall_curve
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold

def pad_or_truncate(docs,max_sents):
	ans = [["<PAD>" for j in range(max_sents)] for i in range(len(docs))]
	for i in range(len(docs)):
		docs[i] = docs[i][:max_sents]
		for k in range(len(docs[i])):
			ans[i][max_sents-len(docs[i])+k] = docs[i][k]
	ans = np.array(ans)
	return ans

file1 = "apwsj_decom_attn_inner_attention_snli_ten_fold_progress.p"
file2 = "apwsj_decom_attn_maxpool_snli_ten_fold_progress.p"
file3 = "apwsj_bilstm_mlp_baseline_ten_fold_progress_baseline.p"
file4 = "apwsj_ablation1_ten_fold_progress.p"
p1,g1,f1,_,_,a1,prb1 = pickle.load(open(file1,"rb"))
p2,g2,f2,_,_,a2,prb2 = pickle.load(open(file2,"rb"))
p3,g3,f3,_,_,prb3 = pickle.load(open(file3,"rb"))
p4,g4,f4,_,_,a4,prb4 = pickle.load(open(file4,"rb"))

targets,sources,target_vecs,source_vecs,gold = pickle.load(open("apwsj_data_maxpool_snli.p","rb"))
print("All files loaded !!!")
max_sents = 336
targets = pad_or_truncate(targets,max_sents)
sources = pad_or_truncate(sources,max_sents)
print("target and source made")
gold_list = [i for i in gold]
kfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 9274)
target = []
source = []
for train,test in kfold.split(np.zeros(len(gold_list)),gold_list):
	target.append(targets[test])
	source.append(sources[test])
print("Target and source really made")
os.remove(file1)
os.remove(file2)
os.remove(file3)
os.remove(file4)
print("Writing new files")
pickle.dump([p1,g1,f1,target,source,a1,prb1],open(file1,"wb"))
pickle.dump([p2,g2,f2,target,source,a2,prb2],open(file2,"wb"))
pickle.dump([p3,g3,f3,target,source,a3,prb3],open(file3,"wb"))
pickle.dump([p4,g4,f4,target,source,prb4],open(file4,"wb"))

sys.exit()

pr1,re1,_ = precision_recall_curve(g1,[i[0] for cv in prb1 for i in cv])
pr2,re2,_ = precision_recall_curve(g2,[i[0] for cv in prb2 for i in cv])
pr3,re3,_ = precision_recall_curve(g3,[i[0] for cv in prb3 for i in cv])
pr4,re4,_ = precision_recall_curve(g4,[i[0] for cv in prb4 for i in cv])

# p_set_diff,r_set_diff,_ = precision_recall_curve(golds,[i[1] for i in probs_set_diff])
# p_geo_diff,r_geo_diff,_ = precision_recall_curve(golds,[i[1] for i in probs_geo_diff])
# p_tfidf_novelty_score,r_tfidf_novelty_score,_ = precision_recall_curve(golds,[i[1] for i in probs_tfidf_novelty_score])
# p_kl_div,r_kl_div,_ = precision_recall_curve(golds,[i[1] for i in probs_kl_div])
# p_pv,r_pv,_ = precision_recall_curve(golds,[i[1] for i in probs_pv])
# p_bilstm_mlp,r_bilstm_mlp,_ = precision_recall_curve(golds,[i[1] for i in probs_bilstm_mlp])
# p_cnn,r_cnn,_ = precision_recall_curve(golds,[i[1] for i in probs_cnn])

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('APWSJ Precision-Recall curves')

plt.plot(re1,pr1,'-', label = 'SNLI Inner attention encoder + decomposable attention')
plt.plot(re2,pr2,'-', label = 'SNLI Maxpool encoder + decomposable attention')
plt.plot(re3,pr3,'-', label = 'SNLI Inner attention encoder + BiLSTM - MLP')
plt.plot(re4,pr4,'-', label = 'Doc2Vec + decomposable attention')

# plt_set_diff = plt.plot(r_set_diff,p_set_diff,'-', label="Set Difference")
# plt_geo_diff = plt.plot(r_geo_diff,p_geo_diff,'-', label="Geometric Difference")
# plt_tfidf_novelty_score = plt.plot(r_tfidf_novelty_score,p_tfidf_novelty_score,'-', label="IDF novelty score")
# plt_kl_div = plt.plot(r_kl_div,p_kl_div,'-', label="KL divergence ( Lang. model )")
# plt_pv = plt.plot(r_pv,p_pv,'-', label="Paragraph vector")
# plt_bilstm_mlp = plt.plot(r_bilstm_mlp,p_bilstm_mlp,'-', label="BiLSTM + MLP")
# plt_cnn = plt.plot(r_cnn,p_cnn,'-', label="RDV + CNN")
# #handles=[plt_set_diff,plt_geo_diff,plt_tfidf_novelty_score,plt_kl_div,plt_pv,plt_bilstm_mlp,plt_cnn]

plt.legend(loc='best')
# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
plt.show()