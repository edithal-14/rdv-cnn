Novelty Goes Deep. A Deep Neural Solution To Document Level Novelty Detection (COLING 2018)
============================================================================================

ABOUT
-----------
RDV-CNN model for document level novelty detection. Comparision of our model with baselines on three popular datasets:
*TAP-DLND 1.0 (https://arxiv.org/abs/1802.06950)
*Webis-CPC (https://www.uni-weimar.de/en/media/chairs/computer-science-department/webis/data/corpus-webis-cpc-11/)
*APWSJ dataset (http://www.cs.cmu.edu/~yiz/research/NoveltyData/)

REQUIREMENTS
-----------------
* Infersent (https://github.com/facebookresearch/InferSent): Infersent is used for training a sentence encoder on SNLI corpus. Required files are already present in the sentence_encoder directory. A pretrained model is also available in sentence_encoder/encoder directory.

*PyTorch (for training the sentence encoder and inferring sentence embeddings)

*Keras

*Tensorflow (for BiLSTM + MLP Baseline)

*Theano (for RDV-CNN model)


Description of important files in each directory
================================================

## DLND

*make_dlnd_data.py:	Produces pre-trained sentence embeddings for dlnd data, dependency: /novelty/infersent directory must be present, dlnd corpus must be present. Creates a pickle file which contains the sentence embeddings.

*rdv.py:	Produces Relative document matrix based on sentence embeddings for input to CNN , input: name of pickle file which has sentence embeddings.

*process.py:	Takes the rdv file and converts it to format which is suitable for input to CNN program, produces a mr_dlnd.p pickle file

*conv_net_sentences.py:	The most important file, this is the main CNN program, give as command line argument path of mr_dlnd.p file. It creates the output file which has the predictions for each target and source document pair

## WEBIS

*make_dlnd_data.py: Produces pre-trained sentence embeddings for dlnd data, dependency: /novelty/infersent directory must be present, dlnd corpus must be present. Creates a pickle file which contains the sentence embeddings.

*rdv.py: Produces Relative document matrix based on sentence embeddings for input to CNN , input: name of pickle file which has sentence embeddings.

*process.py: Takes the rdv file and converts it to format which is suitable for input to CNN program, produces a mr_dlnd.p pickle file

*conv_net_sentences.py: The most important file, this is the main CNN program, give as command line argument path of mr_dlnd.p file. It creates the output file which has the predictions for each target and source document pair

## APWSJ

*make_sentence_embedding.py: Produces pre-trained sentence embeddings for documents in apwsj_parsed_documents directory, dependency: /novelty/infersent directory must be present, apwsj_parsed_documents directory must be present. Creates a pickle file which contains the sentence embeddings.

*make_rdvs.py: Generates Relative document matrix (rdv file ) based on sentence embeddings for input to CNN , input: name of pickle file which has sentence embeddings, output is rdv file

*process.py: It converts the rdv file to format which is suitable for input to CNN program, produces a mr_apwsj.p pickle file

*conv_net_sentences.py: The most important file, this is the main CNN program, give as command line argument path of mr_webis.p file. It creates the output file which has the predictions for each target and source document pair