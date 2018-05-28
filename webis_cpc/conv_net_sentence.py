"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import pickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(datasets,
                   U,
                   img_w, 
                   filter_hs,
                   hidden_units, 
                   dropout_rate,
                   shuffle_batch,
                   n_epochs, 
                   batch_size, 
                   lr_decay,
                   conv_non_linear,
                   activations,
                   sqr_norm_lim,
                   non_static):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(int(time.time()))
    img_h = len(datasets[0][0])-1  
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print(parameters)   
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in range(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(int(time.time()))
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = int(new_data.shape[0]/batch_size)
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = datasets[1][:,:img_h] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)     
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred,test_y_probs = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], [test_error,test_y_pred,test_y_probs] , allow_input_downcast = True)
    
    #start training over mini-batches
    print('... training')
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in range(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        # print "Test losses"
        # for i in range(n_train_batches):
        # 	print test_model(i)
        train_losses = [test_model(i) for i in range(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        # print "Validation losses"
        # for i in range(n_train_batches):
        # 	print val_model(i)
        val_losses = [val_model(i) for i in range(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss,test_prediction,test_probs = test_model_all(test_set_x,test_set_y)      
            test_perf = 1- test_loss
    return test_perf,test_probs,test_prediction,test_set_y

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow )
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow )
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name )
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name )
        gparams.append(gp)
    for param, gp in [(params[i],gparams[i]) for i in range(len(params))]:
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def make_idx_data_cv(sentence_index, cv, labels, curr_cv, max_l, k, filter_h=5):
    train, test = [], []
    for i in range(len(sentence_index)):
        x=[]
        pad = filter_h - 1
        for j in range(pad):
            x.append(0)
        for index in sentence_index[i]:
            x.append(index)
        while len(x) < max_l+2*pad:
            x.append(0)
        x.append(labels[i])
        if cv[i]==curr_cv:
            test.append(x)
        else:
            train.append(x)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]
        
if __name__=="__main__":
    print("loading data...")
    filename = sys.argv[1]
    x = pickle.load(open(filename,"rb"))
    sentence_index,sentence_vecs,labels,cv = x[0], x[1], x[2], x[3]
    print("data loaded!")
    exec(open("conv_net_classes.py","r").read())    
    max_length = max([len(doc) for doc in sentence_index])
    if str(type(sentence_vecs))=="<type 'numpy.ndarray'>":
        print("sentence vecs is numpy array")
	sentence_vecs = [list(row) for row in sentence_vecs]
    vec_dim = len(sentence_vecs[0])
    U = np.zeros(shape=(len(sentence_vecs)+1, vec_dim), dtype='float32')
    #U[0] = np.zeros(vec_dim,dtype='float32')
    for i in range(len(sentence_vecs)):
        U[i]= np.asarray(sentence_vecs[i],dtype='float32')
    results = []
    pred_results=[]
    for i in range(0,10):
        datasets = make_idx_data_cv(sentence_index, cv, labels, i, max_l=max_length,k=vec_dim, filter_h=5)
        perf,test_probs,test_prediction,test_set_y = train_conv_net(datasets,
                              U,
                              img_w=vec_dim,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              hidden_units=[200,2],
                              conv_non_linear="relu", 
                              shuffle_batch=True, 
                              n_epochs=20,
                              sqr_norm_lim=9,
                              non_static=True,
                              batch_size=25,
                              activations=[Iden],
                              dropout_rate=[0.5])
        print("cv: " + str(i) + ", perf: " + str(perf))
        results.append(perf)
        pred_results.append([test_probs,test_prediction,test_set_y])
    print(str(np.mean(results)))
    pickle.dump(pred_results,open(sys.argv[1][3:-2]+"_cnn_output.pickle","wb")) 

    golds = [i for cv in pred_results for i in cv[2]]
    predictions = [i for cv in pred_results for i in cv[1]]

    cf = confusion_matrix(golds,predictions,labels=[0,1])
    acc= accuracy_score(golds,predictions)
    p,r,f,_=precision_recall_fscore_support(golds,predictions,labels=[0,1])
    print("\nConfusion matrix:\n"+str(cf))
    print("\nAccuracy: "+str(acc))
    print("\nClass wise precisions: "+str(p))
    print("Class wise recalls: "+str(r))
    print("Class wise fscores: "+str(f))