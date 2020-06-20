Novelty Goes Deep. A Deep Neural Solution To Document Level Novelty Detection (COLING 2018)
============================================================================================

ABOUT
-----------
RDV-CNN model for document level novelty detection. Comparision of our model with baselines on three popular datasets:
* TAP-DLND 1.0 (https://arxiv.org/abs/1802.06950)
    - Download link: http://www.iitp.ac.in/~ai-nlp-ml/resources.html
* Webis-CPC (https://www.uni-weimar.de/en/media/chairs/computer-science-department/webis/data/corpus-webis-cpc-11/)
* APWSJ dataset (http://www.cs.cmu.edu/~yiz/research/NoveltyData/)

REQUIREMENTS
-----------------
* Python 2.7

* Infersent (https://github.com/facebookresearch/InferSent): Infersent is used for training a sentence encoder on SNLI corpus. Required files are already present in the sentence_encoder directory, please use them, dont use the files from the git repo since they are updated and are no longer compatible with our scripts. A pretrained model is also available in sentence_encoder/encoder directory.

* PyTorch (for training the sentence encoder and inferring sentence embeddings)
    - Version: 1.3.0

* Keras (for BiLSTM + MLP Baseline)
    - Version: 2.3.1

* Tensorflow (for Keras backend)
    - Version: 1.14.0

* Theano (for RDV-CNN model)
    - Version: 1.0.0 (Upgrade as necessary if you face any issues)


Description of important files in each directory
================================================

## DLND

* extract_sentence_embedding.py:	Produces pre-trained sentence embeddings for dlnd data, dependency: ../infersent directory must be present, dlnd corpus must be present. Creates a pickle file which contains the sentence embeddings.

* rdv.py:	Produces Relative document matrix based on sentence embeddings for input to CNN , input: name of pickle file which has sentence embeddings, this is hardcoded.

* process.py:	Takes the rdv file and converts it to format which is suitable for input to CNN program, produces a mr_dlnd.p pickle file

* conv_net_sentences.py:	The most important file, this is the main CNN program, give as command line argument path of mr_dlnd.p file. It creates the output file which has the predictions for each target and source document pair

## WEBIS

* webis_data_preprocessing.py: Converts Webis CPC data to .pickle format which contains source and target sentences as well as the gold values.
    - Input:  Webis-CPC-11 directory should be present in the working directory
    - Output: webis_data.pickle

* webis_sentence_embedding.py: Produces sentence embeddings for webis data.
    - Input:  webis_data.pickle should exist in the cwd
    - Output: webis_embeddings_data_{1024/2048}_attn.p

* process.py: Takes the sentence embedding and converts it to format which is suitable for input to CNN program
    - Input:  webis_embeddings_data_{1024/2048}_attn.p
    - Output: mr_webis_1024_attn.p

* webis_baselines.py: Produces class probabilities for various baselines.
    - Input:  webis_data.pickle and doc2vec.bin
    - Output: webis_baselines_class_probs.p

* webis_bilstm_mlp_baseline.py: Runs BiLSTM + MLP model on webis sentence embeddings and evaluates using 10 fold cross validation, saves the result after each cross validation also prints the result after all the cv are done.
    - Input:  webis_embeddings_data_{1024/2048}_attn.p
    - Output: webis_ten_fold_progress_bilstm_mlp_baseline.p

* conv_net_sentences.py: The most important file, this is the main CNN program, give as command line argument path of mr_webis_1024.p file. It creates the output file which has the predictions for each target and source document pair
    - Input   mr_webis_1024.p
    - Output: webis_1024_cnn_output.pickle

* make_prc_curves.py: Analyze the result of various baselines and BiLSTM + MLP method, produces various scores and plots a precision recall curve. Also stores the class probabilities for each technique in a pickle file.
    - Input: webis_1024_cnn_output.pickle, webis_baselines_class_probs.p, webis_ten_fold_progress_bilstm_mlp_baseline.p
    - Output: webis_prc_curves_data.p

* analyze_cnn_output.py: Analyze the output of conv_net_sentences.py to display the results (precision, recall etc..) of the RDV + CNN model
    - Input: webis_1024_cnn_output.pickle

## APWSJ

* make_sentence_embedding.py: Produces pre-trained sentence embeddings for documents in apwsj_parsed_documents directory, dependency: /novelty/infersent directory must be present, apwsj_parsed_documents directory must be present. Creates a pickle file which contains the sentence embeddings.

* make_rdvs.py: Generates Relative document matrix (rdv file ) based on sentence embeddings for input to CNN , input: name of pickle file which has sentence embeddings, output is rdv file

* process.py: It converts the rdv file to format which is suitable for input to CNN program, produces a mr_apwsj.p pickle file

* conv_net_sentences.py: The most important file, this is the main CNN program, give as command line argument path of mr_apwsj.p file. It creates the output file which has the predictions for each target and source document pair
