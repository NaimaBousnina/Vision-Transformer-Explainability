from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import cv2
import os, argparse
###VIT2_0
from baselines.ViT.ViT_LRP import VisionTransformer
from baselines.ViT.ViT_explanation_generator import LRP
from image_iter import FaceDataset
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from utils.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy
from config import get_config
from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from tensorboardX import SummaryWriter
import torch.nn as nn

import math
""" CUDA_VISIBLE_DEVICES='0' python3 -u train_GViT.py -b 10  -w 0 -d retina -n VIT_LRP  -head CosFace --outdir ./results/ViTLRP --warmup-epochs 1 --lr 3e-4 -t lfw

"""

def cosface(outputs,device_id,label,s=64.0, m=0.35):
        cosine=outputs
        phi = cosine - m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()) #b,num_classes
        if device_id != None:
            one_hot = one_hot.cuda(device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

        one_hot.scatter_(1, label.cuda(device_id[0]).view(-1, 1).long(), 1)   #one_hot size b,num_classes #label_size b
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)
        output *= s 
        
        return output

def sface(outputs,device_id,label,s = 64.0, k = 80.0, a = 0.90, b = 1.2):
    cosine=outputs
    output = cosine * s
    # --------------------------- sface loss ---------------------------

    one_hot = torch.zeros(cosine.size())
    if device_id != None:
        one_hot = one_hot.cuda(device_id[0])
    one_hot.scatter_(1, label.view(-1, 1), 1)

    zero_hot = torch.ones(cosine.size())
    if device_id != None:
        zero_hot = zero_hot.cuda(device_id[0])
    zero_hot.scatter_(1, label.view(-1, 1), 0)


    WyiX = torch.sum(one_hot * output, 1)
    with torch.no_grad():
        theta_yi = torch.acos(WyiX / s)
        weight_yi = 1.0 / (1.0 + torch.exp(-k * (theta_yi - a)))
    intra_loss = - weight_yi * WyiX

    Wj = zero_hot * output
    with torch.no_grad():
        # theta_j = torch.acos(Wj)
        theta_j = torch.acos(Wj / s)
        weight_j = 1.0 / (1.0 + torch.exp(k * (theta_j - b)))
    inter_loss = torch.sum(weight_j * Wj, 1)

    loss = intra_loss.mean() + inter_loss.mean()
    Wyi_s = WyiX / s
    Wj_s = Wj / s
    return output, loss, intra_loss.mean(), inter_loss.mean(), Wyi_s.mean(), Wj_s.mean()

def arcface(outputs,label,device_id, s=64.0, m=0.50, easy_margin=False):
    cosine=outputs

    cos_m = math.cos(m)
    sin_m = math.sin(m)
    th = math.cos(math.pi - m)
    mm = math.sin(math.pi - m) * m    
    sine = torch.sqrt(1.0 - torch.pow(cosine, 2)) # sqrt(1-cos^2)
    phi = cosine * cos_m - sine * sin_m
    if easy_margin:
        phi = torch.where(cosine > 0, phi, cosine)
    else:
        phi = torch.where(cosine > th, phi, cosine - mm)
        # --------------------------- convert label to one-hot ---------------------------
    one_hot = torch.zeros(cosine.size()) #lista de scores
    if device_id != None:
        one_hot = one_hot.cuda(device_id[0])
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    output *= s

    return output


def need_save(acc, highest_acc):   #save the highest accuracy
    do_save = False
    save_cnt = 0   
    if acc[0] > 0.98:  #when accuracy of the first target is higher than 0.98
        do_save = True
    for i, accuracy in enumerate(acc): #save when accuracy gets higher for one of the targets
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i]-0.002: #save when for mor than one target the accuracy of the respective target is higher the highest accuracy-0.002
            save_cnt += 1
    if save_cnt >= len(acc)*3/4 and acc[0]>0.99: 
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-w", "--workers_id", help="gpu ids or cpu", default='cpu', type=str)
    parser.add_argument("-e", "--epochs", help="training epochs", default=125, type=int)  #epoch number
    parser.add_argument("-b", "--batch_size", help="batch_size", default=256, type=int)   #batch size
    parser.add_argument("-d", "--data_mode", help="use which database, [casia, vgg, ms1m, retina, ms1mr]",default='ms1m', type=str)  #data_base
    parser.add_argument("-n", "--net", help="which network, ['VIT','VITs','VIT_LRP']",default='VIT_LRP', type=str)  #model
    parser.add_argument("-head", "--head", help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']", default='ArcFace', type=str) #loss
    parser.add_argument("-t", "--target", help="verification targets", default='lfw,talfw,calfw,cplfw,cfp_fp,agedb_30', type=str) #verification_targets
    parser.add_argument("-r", "--resume", help="resume model", default='', type=str)
    parser.add_argument('--outdir', help="output dir", default='', type=str) #output dir

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', #lr,momentum,opt,weight_decay ->create optimizer
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')           #higher the weight decay smaller the weights
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',   #lr
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',  #Learning rate noise is a technique used in deep learning to introduce random variations in the learning rate during training. 
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',  #warmup lr
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')  #min lr

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',  #number of decay epochs the learning rate is multiplied by a factor (often 0.1 or 0.5) every decay_epochs number of epochs.
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',   #number of warmup epochs
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')   #The cooldown_epochs parameter determines the number of epochs that the learning rate scheduler should wait after reducing the learning rate before resuming training at the new learning rate.
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')   #specify the number of epochs to wait before reducing the learning rate if the validation loss does not improve.
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',    # the learning rate is multiplied by a factor (often 0.1 or 0.5) every decay_epochs
                        help='LR decay rate (default: 0.1)')
    args = parser.parse_args()

    #======= hyperparameters & data loaders =======#
    cfg = get_config(args)  #dictionary 

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED) 

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    EVAL_PATH = cfg['EVAL_PATH']
    WORK_PATH = cfg['WORK_PATH'] # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']

    INPUT_SIZE = cfg['INPUT_SIZE']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    NUM_EPOCH = cfg['NUM_EPOCH']

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    print('GPU_ID', GPU_ID)
    TARGET = cfg['TARGET']
    print(TARGET)
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:   #text file to save configuration
        f.write(str(cfg))
    print("=" * 60)
    
    writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True
    
    with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:   #93431,112,112 class number and image size
        NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]   
    assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]  #check INPUT_SIZE=[112, 112]
    
    dataset = FaceDataset(os.path.join(DATA_ROOT, 'train.rec'), rand_mirror=True) #dataset  image, label
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=len(GPU_ID), drop_last=True)
    
    
    vers = get_val_data(EVAL_PATH, TARGET) #for a single target, example lfw -> vers=['lfw',[lfw_images_dataset],[issame]]
    
    
    print("Number of Training Classes: {}".format(NUM_CLASS))
    loss_type = HEAD_NAME
    
    highest_acc = [0.0 for t in TARGET]
    
    BACKBONE_DICT = {'VIT_LRP': VisionTransformer(
                         num_classes = NUM_CLASS,
                         img_size=112,
                         patch_size=8,
                         embed_dim=EMBEDDING_SIZE,
                         depth=20,
                         num_heads=8,
                         drop_rate=0.1,
                     )}


 
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME] #model is chosen based on the argument
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)
    
    LOSS = nn.CrossEntropyLoss()
    
    
    
    OPTIMIZER = create_optimizer(args, BACKBONE)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)
    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)
    
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        #BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)

    #======= train & validation & save checkpoint =======#
    DISP_FREQ = 200 # frequency to display training loss & acc
    VER_FREQ = 100

    batch = 0  # batch index

    losses = AverageMeter()
    top1 = AverageMeter()

     
    BACKBONE.train()  # set to training mode model.train()
    

    for epoch in range(NUM_EPOCH): # start training process
        
        lr_scheduler.step(epoch)

        last_time = time.time() 

        for inputs, labels in iter(trainloader):
            
            #Inputs shape = numberofinputs,3,112,112
            #labels shape= number of inputs
            
            #print(inputs.shape) 
            #print(labels.shape) 
            
            
            
            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()


            outputs,embs= BACKBONE(inputs.float(),Only_embeddings=False) #TENHO QUE ALTERAR VIT_LRP manda como output os scores
            prec1= train_accuracy(outputs.data, labels, topk = (1,)) #need output.data so it won´t be tracked by backpropagation
                
            #print(outputs.shape)  
            #print(embs.shape) 
            
            if loss_type == 'CosFace':
                outputs = cosface(outputs,GPU_ID,labels) 
            if loss_type == 'ArcFace':
                outputs = arcface(outputs,labels,GPU_ID)
            if loss_type == 'SFaceLoss':
                outputs,_, _,_,_,_ = sface(outputs,labels,GPU_ID)
                            
              
            
            loss = LOSS(outputs, labels) #scores ,target

            #print("outputs", outputs, outputs.data)
            # measure accuracy and record loss
            
            
           

            losses.update(loss.data.item(), inputs.size(0))  #average meter (loss.data.item(), batch size) -> updates the average loss for each epoch
            top1.update(prec1.data.item(), inputs.size(0))   #average meter   -> updates the average precision for each epoch


            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()     # zero the parameter gradients
            loss.backward()
            OPTIMIZER.step()   #adjust parameters based on the calculated gradients
            
            # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                epoch_loss = losses.avg   # average loss of epochs
                epoch_acc = top1.avg      # average precision of epochs
                writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)

                batch_time = time.time() - last_time
                last_time = time.time()

                print('Epoch {} Batch {}\t'
                      'Speed: {speed:.2f} samples/s\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, batch + 1, speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                    loss=losses, top1=top1))
                #print("=" * 60)
                losses = AverageMeter()  # DISP_FREQ = 10 frequency to display training loss & acc, reset average meter every 10 epochs
                top1 = AverageMeter()

            if ((batch + 1) % VER_FREQ == 0) and batch != 0 : #perform validation & save checkpoints (buffer for visualization)
                for params in OPTIMIZER.param_groups:
                    lr = params['lr']
                    break
                print("Learning rate %f"%lr)
                print("Perform Evaluation on", TARGET, ", and Save Checkpoints...") #VER_FREQ = 20 perform evaluation of target every 20 epochs
                acc = []  #initialize accuracy
                for ver in vers:  # for every target
                    name, data_set, issame = ver   #example [['lfw'],[data_set],[issame]]=ver
                    accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, data_set, issame) #validation for verification
                    buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
                    print('[%s][%d]XNorm: %1.5f' % (name, batch+1, xnorm))
                    print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch+1, accuracy, std))
                    print('[%s][%d]Best-Threshold: %1.5f' % (name, batch+1, best_threshold))
                    acc.append(accuracy) #append accuracies

                # save checkpoints per epoch
                if need_save(acc, highest_acc):
                    if MULTI_GPU:
                        torch.save(BACKBONE.module.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                    else:
                        torch.save(BACKBONE.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                BACKBONE.train()  # set to training mode

            batch += 1 # batch index