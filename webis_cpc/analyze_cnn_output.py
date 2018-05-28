from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pickle

preds = pickle.load(open("webis_1024_cnn_output.pickle","rb"))
predictions = [j for i in preds for j in i[1]]
golds = [j for i in preds for j in i[2]]
class0_scores = [j[0] for i in preds for j in i[0]]
class1_scores = [j[1] for i in preds for j in i[0]]
del preds

# cf = confusion_matrix(gold,predictions)
# print("Given Non-novel Predicted Non-novel: "+str(cf[0][0]))
# print("Given Non-novel Predicted Novel: "+str(cf[0][1]))
# print("Given Novel Predicted Non-novel: "+str(cf[1][0]))
# print("Given Novel Predicted Novel: "+str(cf[1][1]))
# invert_gold = [1-i for i in gold]
# precision,recall,thresholds = precision_recall_curve(invert_gold,class0_scores)
# #average_precision = average_precision_score(invert_gold, class0_scores)
cf = confusion_matrix(golds,predictions,labels=[0,1])
acc= accuracy_score(golds,predictions)
p,r,f,_=precision_recall_fscore_support(golds,predictions,labels=[0,1])
print("\nConfusion matrix:\n"+str(cf))
print("\nAccuracy: "+str(acc))
print("\nClass wise precisions: "+str(p))
print("Class wise recalls: "+str(r))
print("Class wise fscores: "+str(f))

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# #plt.title('2-class Precision-Recall curve: Average precision={0:0.2f}'.format(average_precision))
# plt.title('Non-novel Precision-Recall curve')
# plt.plot(recall,precision,'b-')
# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
# plt.show()