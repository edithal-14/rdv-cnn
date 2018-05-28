import pickle
import numpy as np
from scipy.spatial.distance import cdist
import xml.etree.ElementTree as ET

def build_rdv(t,s):
	return np.concatenate([t,s,np.subtract(t,s),np.multiply(t,s)],axis=0)

def rdv(target_matrix,source_matrix,target_files,dir_n,doc_n):
	match = np.argmin(cdist(target_matrix,source_matrix,metric="cosine"),axis=1)
	relative_doc_vector = np.vstack((build_rdv(target_matrix[i],source_matrix[match[i]]) for i in range(target_matrix.shape[0])))
	label = [tag.attrib["DLA"] for tag in ET.parse(target_files[dir_n][doc_n][:-4]+".xml").findall("feature") if "DLA" in tag.attrib.keys()][0]
	return [relative_doc_vector,label]

if __name__=="__main__":
	source_docs_vectors,target_docs_vectors,source_docs,target_docs,source_files,target_files=pickle.load(open("sentence_embeddings.pickle","rb"))
	relative_doc_vectors = [rdv(target_docs_vectors[i][j],np.vstack((source_docs_vectors[i][k] for k in range(len(source_docs_vectors[i])))),target_files,i,j) for i in range(len(target_docs_vectors)) for j in range(len(target_docs_vectors[i])-1)]
	pickle.dump(relative_doc_vectors,open("rdvs.pickle","wb"))