import sys
import pickle
import matplotlib.pyplot as plt

filename = sys.argv[1]
novel_class = int(sys.argv[2])

preds = pickle.load(open(filename,"rb"))
gold = [j for i in preds for j in i[2]]
class0_scores = [j[0] for i in preds for j in i[0]]
class1_scores = [j[1] for i in preds for j in i[0]]

if novel_class==1:
	novel_probs = class1_scores
	non_novel_probs = class0_scores
else:
	novel_probs = class0_scores
	non_novel_probs = class1_scores

given_novel_pred_novel = [novel_probs[i] for i in range(len(gold)) if gold[i]==novel_class]
given_novel_pred_non_novel = [non_novel_probs[i] for i in range(len(gold)) if gold[i]==novel_class]
given_non_novel_pred_novel = [novel_probs[i] for i in range(len(gold)) if gold[i]!=novel_class]
given_non_novel_pred_non_novel = [non_novel_probs[i] for i in range(len(gold)) if gold[i]!=novel_class]


plt.subplot(2,2,1)
plt.scatter([i for i in range(len(given_novel_pred_novel))],given_novel_pred_novel,0.5,'b','o')
plt.title("Given Novel, Predicted novel")
plt.ylabel("Probability")
plt.xlabel("Document no.")

plt.subplot(2,2,2)
plt.scatter([i for i in range(len(given_novel_pred_non_novel))],given_novel_pred_non_novel,0.5,'r','o')
plt.title("Given Novel, Predicted Non-novel")
plt.ylabel("Probability")
plt.xlabel("Document no.")

plt.subplot(2,2,3)
plt.scatter([i for i in range(len(given_non_novel_pred_novel))],given_non_novel_pred_novel,0.5,'c','o')
plt.title("Given Non-novel, Predicted Novel")
plt.ylabel("Probability")
plt.xlabel("Document no.")

plt.subplot(2,2,4)
plt.scatter([i for i in range(len(given_non_novel_pred_non_novel))],given_non_novel_pred_non_novel,0.5,'g','o')
plt.title("Given Non-novel, Predicted Non-novel")
plt.ylabel("Probability")
plt.xlabel("Document no.")

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()
