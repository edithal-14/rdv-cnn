from nltk import sent_tokenize,word_tokenize
import os
import pickle

if __name__=="__main__":
	webis_path="Webis-CPC-11/"
	n_cases = len(os.listdir(webis_path))/3
	webis_data = [[sent_tokenize(open(webis_path+str(i+1)+"-original.txt","r").read()),sent_tokenize(open(webis_path+str(i+1)+"-paraphrase.txt","r").read()),1 if open(webis_path+str(i+1)+"-metadata.txt","r").read().splitlines()[3].split(" ")[1]=="Yes" else 0] for i in range(n_cases) ]
	print(n_cases)
	pickle.dump(webis_data,open("webis_data.pickle","wb"))