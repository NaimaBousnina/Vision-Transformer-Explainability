import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from .verification import evaluate

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import mxnet as mx
import io
import os, pickle, sklearn
import time
from IPython import embed

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def load_bin(path, image_size=[112,112]):
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes') #issame_list list of genuine and impostor pairs
    data_list = [] 
    print(len(bins))
    print(len(issame_list))

    for flip in [0,1]:
        data = torch.zeros((len(issame_list)*2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list)*2): #12000
        _bin = bins[i]  
        img = mx.image.imdecode(_bin) #convert to image
        if img.shape[1]!=image_size[0]: 
            img = mx.image.resize_short(img, image_size[0])
        img = mx.nd.transpose(img, axes=(2, 0, 1))  #channels,h,w
        for flip in [0,1]:
            if flip==1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = torch.tensor(img.asnumpy()) #datalist[0][i][channel][h][w] ,datalist[1][i][channel][h][w]
        if i%1000==0:
            print('loading bin', i)
    print(data_list[0].shape)  # size data_list.shape = 2,len(issame_list*2),3,112,112
    return data_list, issame_list  #images dataset , issame_list -> boolean list 


def get_val_pair(path, name):
    ver_path = os.path.join(path,name + ".bin") # example "./eval/lfw.bin"
    print(ver_path) 
    assert os.path.exists(ver_path) #check if exists
    data_set, issame = load_bin(ver_path)  #data_set images no-flip and flip 
    print('ver', name)
    return data_set, issame


def get_val_data(data_path, targets):  # data path :'./eval/' , targets:lfw,talfw,calfw,cplfw,cfp_fp,agedb_30 
    assert len(targets) > 0 #check number of targets
    vers = [] #list
    for t in targets: #for t in lfw,talfw,calfw,cplfw,cfp_fp,agedb_30
        data_set, issame = get_val_pair(data_path, t) #get validation pair 
        vers.append([t, data_set, issame]) # example ['lfw',[data_set],[issame]]
    return vers


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def separate_mobilefacenet_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'mobilefacenet' in str(layer.__class__) or 'container' in str(layer.__class__):
            continue
        if 'batchnorm' in str(layer.__class__):
            paras_only_bn.extend([*layer.parameters()])
        else:
            paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    #plt.savefig("roc_curve.png")
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
   
    buf.seek(0)
    plt.close()

    return buf

def test_forward(device, backbone, data_set):
    backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode
    #embed()
    #last_time1 = time.time()
    forward_time = 0
    carray = data_set[0]
        #print("carray:",carray.shape)
    idx = 0
    with torch.no_grad():
            while idx < 2000:
                batch = carray[idx:idx + 1]
                batch_device = batch.to(device)
                last_time = time.time()
                backbone(batch_device)
                forward_time += time.time() - last_time
                #if idx % 1000 ==0:
                #    print(idx, forward_time)
                idx += 1
    print("forward_time", 2000, forward_time, 2000/forward_time)
    return forward_time

def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode 

    embeddings_list = []  #embedding list
    #data_set.shape = 2,len(issame_list*2),3,112,112
    for carray in data_set: #[[data_set_flip].shape,[data_set_noflip].shape]->[[12000,3,112,112],[120000,3,112,112]]
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size]) # 12000, embedding size
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx:idx + batch_size]
                #last_time = time.time()
                embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu() # only outputs the embeddings, not the scores (label=None)
                #batch_time = time.time() - last_time    #MANDAR EMBEDDINGS
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray): # ['''''''''''''''''''(idx)........] some face embeddings were not computed
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
            embeddings_list.append(embeddings)
    #embeddings_list.shape -> [2,12000,embedding_size]
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds,_ = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def perform_val_without_flip(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode 

    embeddings_list = []  #embedding list
    #data_set.shape = 2,len(issame_list*2),3,112,112
     #[[data_set_flip].shape,[data_set_noflip].shape]->[[12000,3,112,112],[120000,3,112,112]]
    idx = 0
    embeddings = np.zeros([len(data_set), embedding_size]) # 12000, embedding size
    with torch.no_grad():
        while idx + batch_size <= len(data_set):
            batch = data_set[idx:idx + batch_size]
            #last_time = time.time()
            embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu() # only outputs the embeddings, not the scores (label=None)
            #batch_time = time.time() - last_time    #MANDAR EMBEDDINGS
            #print("batch_time", batch_size, batch_time)
            idx += batch_size
        if idx < len(data_set): # ['''''''''''''''''''(idx)........] some face embeddings were not computed
            batch = data_set[idx:]
            embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)
    
    
    
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0]
    embeddings = sklearn.preprocessing.normalize(embeddings)

    tpr, fpr, accuracy, best_thresholds,dist = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor,dist,embeddings

def perform_val_without_flip_dataloader(multi_gpu, device, embedding_size, backbone, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode 

    embeddings_list = []  #embedding list
    batch_size=data_set.batch_size
    #print(batch_size)
    #data_set.shape = 2,len(issame_list*2),3,112,112
     #[[data_set_flip].shape,[data_set_noflip].shape]->[[12000,3,112,112],[120000,3,112,112]]
    idx = 0
    embeddings = np.zeros([len(data_set)*2, embedding_size]) # 12000, embedding size
    #print(len(data_set)*2)
    with torch.no_grad():
        for batch in data_set:
            if batch.shape[0]==batch_size:
                #last_time = time.time()
                embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu() # only outputs the embeddings, not the scores (label=None)
                #batch_time = time.time() - last_time    #MANDAR EMBEDDINGS
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
            else: # ['''''''''''''''''''(idx)........] some face embeddings were not computed
                embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)
    #embeddings_list.shape -> [2,12000,embedding_size]
    
    
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0]
    embeddings = sklearn.preprocessing.normalize(embeddings)

    tpr, fpr, accuracy, best_thresholds,dist = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor,dist,embeddings,tpr,fpr

def perform_val_without_flip_Multiple_ViTs_and_ORIGINAL_VIT_dataloader(multi_gpu, device, embedding_size, backbone, data_set_VIT,data_set_Mulitple_VITs, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode 
    if len(data_set_VIT) != len(data_set_Mulitple_VITs):
        raise ValueError("The lengths of the two datasets are not the same")
    embeddings_list = []  #embedding list
    #data_set.shape = 2,len(issame_list*2),3,112,112
     #[[data_set_flip].shape,[data_set_noflip].shape]->[[12000,3,112,112],[120000,3,112,112]]
    idx = 0
    batch_size=data_set_Mulitple_VITs.batch_size
    embeddings = np.zeros([len(data_set_VIT)*2, embedding_size]) # 12000, embedding size
    combined_iterator = zip(iter(data_set_VIT),iter(data_set_Mulitple_VITs))

    with torch.no_grad():  
        for batch1, batch2 in combined_iterator:
            if batch2.shape[0]==batch_size:
                #batch1 = data_set_VIT[idx:idx + batch_size]
                #last_time = time.time()
                embeddings[idx:idx + batch_size] = backbone(batch1.to(device),batch2.to(device)).cpu() # only outputs the embeddings, not the scores (label=None)
                #batch_time = time.time() - last_time    #MANDAR EMBEDDINGS
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
            else:
                #batch1 = data_set_VIT[idx:]
                # ['''''''''''''''''''(idx)........] some face embeddings were not computed
                embeddings[idx:] = backbone(batch1.to(device),batch2.to(device)).cpu()
        
        embeddings_list.append(embeddings)
    #embeddings_list.shape -> [2,12000,embedding_size]
    
    
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0]
    embeddings = sklearn.preprocessing.normalize(embeddings)

    tpr, fpr, accuracy, best_thresholds,dist = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor,dist,embeddings,tpr,fpr












def perform_val_deit(multi_gpu, device, embedding_size, batch_size, backbone, dis_token, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
   
    for carray in data_set:
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx:idx + batch_size]
                #last_time = time.time()
                #embed()
                fea,token = backbone(batch.to(device), dis_token.to(device))
                embeddings[idx:idx + batch_size] = fea.cpu()
                #batch_time = time.time() - last_time
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray):
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds,_ = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def buffer_val(writer, db_name, acc, std, xnorm, best_threshold, roc_curve_tensor, batch):
    writer.add_scalar('Accuracy/{}_Accuracy'.format(db_name), acc, batch)
    writer.add_scalar('Std/{}_Std'.format(db_name), std, batch)
    writer.add_scalar('XNorm/{}_XNorm'.format(db_name), xnorm, batch)
    writer.add_scalar('Threshold/{}_Best_Threshold'.format(db_name), best_threshold, batch)
    writer.add_image('ROC/{}_ROC_Curve'.format(db_name), roc_curve_tensor, batch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val    #val
        self.sum   += val * n    
        self.count += n
        self.avg   = self.sum / self.count

'''
def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
'''

def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk) # only considers the first top score (i can consider top-5 for example)
    batch_size = target.size(0)   #number of labels, same as batch size

    _, pred = output.topk(maxk, 1, True, True)  #returns values, indices #outputs the label obtained by the model
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)) #returns a tensor of booleans
    #embed()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)   #False -> 0 True -> 1 sum the number of correct labels
        res.append(correct_k.mul_(100.0 / batch_size))    #percentage of correct labels -> precision

    return res[0] #top 1 precision