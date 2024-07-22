from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import cv2
import os, argparse

from baselines.ViT.ViT_LRP import VisionTransformer
from baselines.ViT.ViT_explanation_generator import LRP
from image_iter import FaceDataset
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from utils.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy,perform_val_without_flip_dataloader
from Config_Multiple_ViTs import get_config
from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import torch.nn as nn
import math
from utils.utils import perform_val_without_flip_Multiple_ViTs_and_ORIGINAL_VIT_dataloader

dtype = np.float32

def gaussian_kernel(size, sigma, type='Sum'):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1,
           -size // 2 + 1:size // 2 + 1]
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    #print("kernel shape {}".format(kernel.shape))
    if type=='Sum':
      kernel = kernel / kernel.sum()
    else:
      kernel = kernel / kernel.max()
    return kernel.astype('double')







def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #cv2.imwrite(str(i)+'converted.jpg',heatmap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization_VIT(attribution_generator ,original_image,use_thresholding =  False, class_index=None,method="transformer_attribution",is_ablation=False,head_fusion="mean",discard_ratio=0.9,EBM=None,EBM_emb_scores=None,histogram=False,Specific_Area=0):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method=method, index=class_index,is_ablation=is_ablation,head_fusion=head_fusion,discard_ratio=discard_ratio,EBM=EBM,EBM_emb_scores=EBM_emb_scores).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    #print(transformer_attribution.shape)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=8, mode='bilinear')  #14->224
    #print(transformer_attribution.shape)
    transformer_attribution = transformer_attribution.reshape(112, 112).data.cpu().numpy()  #/(224,224)
    attention_map= transformer_attribution.copy()
    #print(transformer_attribution.shape)
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
        transformer_attribution = transformer_attribution * 255
        transformer_attribution = transformer_attribution.astype(np.uint8)
        ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        transformer_attribution[transformer_attribution == 255] = 1
        

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    #print(image_transformer_attribution.shape)
    if not np.all(image_transformer_attribution == 0):
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_BGR2RGB)
    return vis,transformer_attribution,attention_map


def Explainability_heatmap(attribution_generator,pair,method="transformer_attribution",is_ablation=False,head_fusion="mean",discard_ratio=0.9):
    _,transformer_attribution1_original,_=generate_visualization_VIT(attribution_generator,pair[0],method=method,head_fusion=head_fusion,is_ablation=is_ablation)
    _,transformer_attribution1,_=generate_visualization_VIT(attribution_generator,pair[0],True,method=method,head_fusion=head_fusion,is_ablation=is_ablation)
    _,transformer_attribution2_original,_=generate_visualization_VIT(attribution_generator,pair[1],method=method,head_fusion=head_fusion,is_ablation=is_ablation)
    _,transformer_attribution2,_=generate_visualization_VIT(attribution_generator,pair[1],True,method=method,head_fusion=head_fusion,is_ablation=is_ablation)



    A = np.logical_xor(transformer_attribution1, transformer_attribution2).astype(int)
    B=np.logical_or(transformer_attribution1, transformer_attribution2).astype(int)
    And_matrix=np.logical_and(transformer_attribution1, transformer_attribution2).astype(int)
    
    U_I_heatmap=(transformer_attribution1*transformer_attribution1_original)*A+(transformer_attribution2*transformer_attribution2_original)*A
    U_heatmap=((transformer_attribution1*transformer_attribution1_original)*And_matrix+(transformer_attribution2*transformer_attribution2_original)*And_matrix)/2 +U_I_heatmap
    Intersection_heatmap=((transformer_attribution1*transformer_attribution1_original)*And_matrix+(transformer_attribution2*transformer_attribution2_original)*And_matrix)/2
    
    U_I_heatmap = (U_I_heatmap - U_I_heatmap.min()) / (U_I_heatmap.max() - U_I_heatmap.min())
    U_heatmap = (U_heatmap - U_heatmap.min()) / (U_heatmap.max() - U_heatmap.min())
    Intersection_heatmap=(Intersection_heatmap -Intersection_heatmap.min()) / (Intersection_heatmap.max() - Intersection_heatmap.min())

    U_heatmap_copy=U_heatmap.copy()
    U_I_heatmap_copy=U_I_heatmap.copy()
    Intersection_heatmap_copy=Intersection_heatmap.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))  # Adjust the kernel size as needed #CHANGE
    
    U_I_heatmap = cv2.dilate(U_I_heatmap, kernel, iterations=1)
    U_heatmap = cv2.dilate(U_heatmap, kernel, iterations=1)
    Intersection_heatmap=cv2.dilate(Intersection_heatmap, kernel, iterations=1)
    
    
   
    
   
    Kernel_size=56 #16,4
    hm = gaussian_kernel(Kernel_size,Kernel_size/8.5)  #original64,8.5  #new filter 8,8 dilation filter  16,16/4 #  8,8 dilation filter 56,56/8.5  # 16,16 16,16/4
    X  = cv2.filter2D( U_I_heatmap,-1,hm)
    U_I_heatmap = X-np.min(X)
    U_I_heatmap = (U_I_heatmap - U_I_heatmap.min()) / (U_I_heatmap.max() - U_I_heatmap.min())
    
    
    X  = cv2.filter2D(U_heatmap,-1,hm)
    U_heatmap = X-np.min(X)
    U_heatmap = (U_heatmap - U_heatmap.min()) / (U_heatmap.max() - U_heatmap.min())



    X  = cv2.filter2D(Intersection_heatmap,-1,hm)
    Intersection_heatmap = X-np.min(X)
    Intersection_heatmap = (Intersection_heatmap - Intersection_heatmap.min()) / (Intersection_heatmap.max() - Intersection_heatmap.min())


           
    IM_heatmap2=transformer_attribution2_original
    IM_heatmap2 = cv2.dilate(IM_heatmap2, kernel, iterations=1)
    X  = cv2.filter2D(IM_heatmap2,-1,hm)
    IM_heatmap2 = X-np.min(X)
    IM_heatmap2 = (IM_heatmap2 - IM_heatmap2.min()) / (IM_heatmap2.max() - IM_heatmap2.min())
    

    IM_heatmap1=transformer_attribution1_original
    IM_heatmap1 = cv2.dilate(IM_heatmap1, kernel, iterations=1)
    X  = cv2.filter2D(IM_heatmap1,-1,hm)
    IM_heatmap1 = X-np.min(X)
    IM_heatmap1 = (IM_heatmap1 - IM_heatmap1.min()) / (IM_heatmap1.max() - IM_heatmap1.min())
    

    return Intersection_heatmap,IM_heatmap1,IM_heatmap2,U_heatmap,U_I_heatmap,transformer_attribution1_original, transformer_attribution2_original,U_heatmap_copy, U_I_heatmap_copy, Intersection_heatmap_copy








def generate_visualization_LVITs(attribution_generator ,original_image,use_thresholding =  False, class_index=None,method="transformer_attribution",is_ablation=False,head_fusion="mean",discard_ratio=0.9,EBM=None,EBM_emb_scores=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0), method=method, index=class_index,is_ablation=is_ablation,head_fusion=head_fusion,discard_ratio=discard_ratio,EBM=EBM,EBM_emb_scores=EBM_emb_scores).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 4, 4)
    #print(transformer_attribution.shape)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=(28/4), mode='bilinear')  #14->224
    #print(transformer_attribution.shape)
    transformer_attribution = transformer_attribution.reshape(28, 28).data.cpu().numpy()
    #/(224,224)
    attention_map= transformer_attribution.copy()
    #print(transformer_attribution.shape)
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
        transformer_attribution = transformer_attribution * 255
        transformer_attribution = transformer_attribution.astype(np.uint8)
        ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        transformer_attribution[transformer_attribution == 255] = 1
        
        
    

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    #print(image_transformer_attribution.shape)
    if not np.all(image_transformer_attribution == 0):
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_BGR2RGB)
    return vis,transformer_attribution, image_transformer_attribution,attention_map













def tensor_to_image(image_transformer_attribution):
    tensor= (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def calculate_accuracy(threshold, dist, actual_issame): #calculate accuracy based on the threshold
    predict_issame = np.less(dist, threshold)
   
    tp = np.sum(np.logical_and(predict_issame, actual_issame)) #true positives
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame))) #false positives
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    tnr = 0 if (tp + fn == 0) else float(tn) / float(tn + fp)
    fnr = 0 if (fp + tn == 0) else float(fn) / float(fn + tp)

    
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc,tnr,fnr



import struct

from torch.utils.data import Dataset

#Original dataset dataloader

def Original_read_test(filename,image_index):
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    # Calculate the pair index and the image position within the pair
    pair_index = image_index // 2
    is_first_image = image_index % 2 == 0

    # Calculate the byte offsets for the specific pair
    boolean_offset = pair_index * ( np.dtype(np.uint8).itemsize + 3*112*112*2 * itemsize)
    output_offset = boolean_offset + np.dtype(np.uint8).itemsize
    output2_offset = output_offset + 3*112*112*itemsize
   

    # Open the binary file for reading
    with open(filename, "rb") as fin:
        # Read the boolean value, output1,, output2,  for the specific pair
        # Move the file pointer to the start of the specific pair
        fin.seek(boolean_offset)

        # Read the boolean value
        boolean_value = np.fromfile(fin, dtype=np.uint8, count=1)
        boolean_value =bool(boolean_value[0])

        if is_first_image:
            output1 = np.fromfile(fin, dtype=dtype, count=3*112*112).reshape((3, 112, 112))
     
            return output1
        else:
            fin.seek(output2_offset)
            output2 = np.fromfile(fin, dtype=dtype, count=3*112*112).reshape((3, 112, 112))

            return output2
def Original_read_pair(filename, pair_index):
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    # Calculate the byte offsets for the specific pair
    boolean_offset = pair_index * (np.dtype(np.uint8).itemsize +  3 * 112 * 112 * 2 * itemsize )
    output_offset = boolean_offset + np.dtype(np.uint8).itemsize
    output2_offset = output_offset +  3 * 112 * 112 * itemsize
    

    # Open the binary file for reading
    with open(filename, "rb") as fin:
        # Move the file pointer to the start of the specific pair
        fin.seek(output_offset)

        # Read output1 and output2 for the specific pair
        output1 = np.fromfile(fin, dtype=dtype, count= 3 * 112 * 112).reshape(( 3, 112, 112))
        fin.seek(output2_offset)
        output2 = np.fromfile(fin, dtype=dtype, count= 3 * 112 * 112).reshape((3, 112, 112))

    # Concatenate output1 and output2
    unsqueezed_output1 = np.expand_dims(output1, axis=0)
    unsqueezed_output2 = np.expand_dims(output2, axis=0)
    concatenated_output = np.concatenate((unsqueezed_output1, unsqueezed_output2), axis=0)
    concatenated_output=torch.from_numpy(concatenated_output)
    return concatenated_output

def Original_read_batch(filename,num_pairs,batch_size):
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
   
    batch = torch.tensor([])
    with open(filename, "rb") as fin:
        for i in range(num_pairs):
            boolean_value = np.fromfile(fin, dtype=np.uint8, count=1)
            boolean_value =bool(boolean_value[0])

            output1 = np.fromfile(fin, dtype=dtype, count=3*112*112).reshape((3, 112, 112))
            output2 = np.fromfile(fin, dtype=dtype, count=3*112*112).reshape(( 3, 112, 112))

            sample = torch.from_numpy(output1).unsqueeze(0)
            batch = torch.cat((batch, sample), dim=0)
            sample = torch.from_numpy(output2).unsqueeze(0)
            batch = torch.cat((batch, sample), dim=0)
            
           
            if batch.shape[0] == batch_size:
                yield batch
                batch = torch.tensor([])
              
        if batch.shape[0]> 0:
            yield batch
            batch = torch.tensor([])


def Original_read_boolean(filename,pair_index):
    itemsize = np.dtype(dtype).itemsize

    # Calculate the pair index and the image position within the pair
   

    # Calculate the byte offsets for the specific pair
    boolean_offset = pair_index * ( np.dtype(np.uint8).itemsize + 3*112*112*2* itemsize )
    with open(filename, "rb") as fin:
        # Read the boolean value, output1, landmarks1, output2, and landmarks2 for the specific pair
        # Move the file pointer to the start of the specific pair
        fin.seek(boolean_offset)

        # Read the boolean value
        boolean_value = np.fromfile(fin, dtype=np.uint8, count=1)
        boolean_value =bool(boolean_value[0])
        
    return boolean_value

class TestDataset_Original(Dataset):
    def __init__(self, filename, start=None, end=None, dtype=np.float32,batch_size=10):
        super().__init__()
        self.filename = filename
        self.start = start
        self.end = end
        self.dtype = dtype
        self.batch_size=batch_size
        

    def __len__(self):   #Number of pairs
        if self.start is not None and self.end is not None:
            return (self.end - self.start)


    def __getitem__(self, index):
        if self.start is not None:
            index += self.start
        if index >= self.end*2: #self.end number of pairs
            raise IndexError("Index out of range")

        sample = Original_read_test(self.filename, index)

        sample = torch.from_numpy(sample)

        return  sample
    
    def read_pair_outputs(self,pair_index):
        return Original_read_pair(self.filename, pair_index)
    

    
    
    def boolean(self,pair_index):
        return Original_read_boolean(self.filename,pair_index)
    
    
    def __iter__(self):
        return  Original_read_batch(self.filename, len(self), batch_size=self.batch_size)       


from minplus_utils import read_img,imshow,heatmap,contours,color_mask,show_superpixels
from minplus_utils import saliency_LIME_masked_areas_score,saliency_minus,saliency_plus,saliency_RISE,saliency_LIME,saliency_LIME_Multiple_ViTs
import sklearn

def extract_regions_from_image(Landmarks, original_image):
    regions_tensor = torch.zeros(10, 3, 28, 28)
    y1, x1 = map(int, Landmarks[0])
    y2, x2 = map(int, Landmarks[1])
    ynose, xnose = map(int, Landmarks[6])

    for i in range(10):
        y, x = Landmarks[i].astype(int)
        region_img = original_image[:, x:x+28, y:y+28]

        if y1 == y2 and x1 == x2 and ynose>y1 and (i in [3, 7, 1]):
           region_img = region_img[:, :, ::-1]                 
            
           
        if y1 == y2 and x1 == x2 and ynose<y1 and (i in [0, 2, 5]):
           region_img = region_img[:, :, ::-1]                 
            
            
        regions_tensor[i] = region_img

    return regions_tensor

def mask_original_image(landmarks_to_mask,Landmarks_image,image):
    copy_image =image.clone()

    for i in landmarks_to_mask:
            y, x = Landmarks_image[i].astype(int)
            copy_image[:, x+14:x+28, y:y+28]=torch.zeros(3, 14, 28)

    #imshow(tensor_to_image(copy_image.permute((1,2,0))),fpath='masked_Image.png')    
    #tensor=extract_regions_from_image(Landmarks_image, copy_image)
    
    #imshow(Image_with_patches(Landmarks_image,tensor,[0,1,2,3,4,5,6,7,8,9]),fpath='masked_Image_patches.png') 
    return copy_image



def LIME_Original(probe,gallery,model):
    pos     = (15,25)
    def fx(X):
            # y is the reference embedding
        
        
        X_tensor = torch.from_numpy(X) 
        x = model(X_tensor.float().permute((2,0,1)).unsqueeze(0).cuda()).cpu()
        # Move tensor from GPU to CPU
        x_cpu = x.detach()

        # Convert tensor to NumPy array
        x_np = x_cpu.numpy()
        
        x_np = np.squeeze(x_np)
        

        # Perform the dot product
        score = np.dot(x_np, y)
        return score
    
    Y=gallery.permute((1,2,0))
    X=probe.permute((1,2,0))    
    y=model(Y.permute((2,0,1)).unsqueeze(0).cuda()).squeeze().cpu().detach()
    
    Xsp,Ms,Z,superpixels = saliency_LIME(X.numpy(),fx)
    Xk = color_mask(X.numpy().astype(np.uint8),Ms,[64,64,64])
    Xss = show_superpixels(X.numpy().astype(np.uint8),superpixels)   
    #print(Xk)
    #print(Xss)
    imshow(Z ,fpath='heatmap_LIME_without_image.png')
    D1,Ylime = heatmap(X.numpy().astype(np.uint8),Z,97)
    #X_image=cv2.cvtColor(X.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    #Y_image=cv2.cvtColor(Y.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    #Xk=cv2.cvtColor(Xk, cv2.COLOR_RGB2BGR)
    #Xss=cv2.cvtColor(Xss, cv2.COLOR_RGB2BGR)
    
    
    #Xin       = cv2.hconcat([Y_image,X_image])
    #cv2.putText(Ylime,'LIME', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    
    #cv2.putText(Xss,'Superpixels', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    #cv2.putText(Xk,'Mask', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    #Ilime = cv2.hconcat([Xin ,Xss,Xk,Ylime])
    #Ilime = cv2.hconcat([ Xin ,Ylime])
    #imshow(Ilime,fpath='heatmap_LIME.png')
    return Ylime

def MinPlus_Original(probe,gallery,model):
    def fx_face_matching(X):
            # y is the reference embedding
        
        
        X_tensor = torch.from_numpy(X) 
        x = model(X_tensor.float().permute((2,0,1)).unsqueeze(0).cuda()).cpu()
        # Move tensor from GPU to CPU
        x_cpu = x.detach()

        # Convert tensor to NumPy array
        x_np = x_cpu.numpy()
        
        x_np = np.squeeze(x_np)
        

        # Perform the dot product
        score = np.dot(x_np, y)
        return score
    

    # 4) PARAMETERS OF MINPLUS

    FAST_MODE = False

    if FAST_MODE:  # 1 minute
        gsigma    = 221                # width of Gaussian mask
        d         = 48                 # steps (one evaluation each d x d pixeles)
        tmax      = 1                  # maximal number of iterations
    else:          # 6 minutes
        gsigma    = 161                # width of Gaussian mask
        d         = 16                  # steps (one evaluation each d x d pixeles)
        tmax      = 20                 # maximal number of iterations
        dsc         = 0.01
        fx          = fx_face_matching
    
    Y=gallery.permute((1,2,0))
    X=probe.permute((1,2,0))    
    y=model(Y.permute((2,0,1)).unsqueeze(0).cuda()).squeeze().cpu().detach()
    H0m,H1m = saliency_minus(X.numpy(),fx,nh=gsigma,d=d,n=tmax,nmod=2,th=dsc)
    S0m,Y0m = heatmap(X.numpy().astype(np.uint8),H0m,gsigma)
    S1m,Y1m = heatmap(X.numpy().astype(np.uint8),H1m,gsigma)
    
    
    #pos     = (15,25)
    #cv2.putText(Y0m,'S0-', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    #cv2.putText(Y1m,'S1-', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    X_image=cv2.cvtColor(X.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    Y_image=cv2.cvtColor(Y.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    Xin       = cv2.hconcat([Y_image,X_image])
    Im      = cv2.hconcat([Xin,Y0m,Y1m])
    #imshow(Im,fpath='heatmap_minus.png',show_pause=1)
    
    
    H0p,H1p   = saliency_plus(X.numpy(),fx,nh=gsigma,d=d,n=tmax,nmod=1,th=dsc)
    S0p,Y0p   = heatmap(X.numpy().astype(np.uint8),H0p,gsigma)
    S1p,Y1p   = heatmap(X.numpy().astype(np.uint8),H1p,gsigma)
    
    #cv2.putText(Y0p,'S0+', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    #cv2.putText(Y1p,'S1+', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    #Ip        = cv2.hconcat([Xin,Y0p,Y1p])
    #imshow(Ip,fpath='heatmap_plus.png',show_pause=1)
    
    
    Havg      = (H0m+H0p+H1m+H1p)/4 # <= HeatMap between 0 and 1
    print("Havg shape {}".format(Havg.shape ))
    Smp,Ymp   = heatmap(X.numpy().astype(np.uint8),Havg,gsigma)
    #imshow( Smp ,fpath='heatmap_MinPlus_Smpwithout_image.png')
    
    #cv2.putText(Ymp,'MinPlus', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    C         = contours(255*Smp, X_image,'jet',10,print_levels=False,color_levels=True)
    Imp       = cv2.hconcat([Xin,Ymp,C ])
    
    imshow(Imp,fpath='heatmap_MinPlus_GViT.png',show_pause=1) 
    
    #print(Smp)
    return Ymp,Smp






def concatenate_tensors(tensor1, tensor2):
    # Check if tensors have the correct size (3, 112, 112)
    if tensor1.size() != tensor2.size() :
        raise ValueError("Both input tensors must be of same size")

    # Add a new dimension to both tensors to make them (1, 3, 112, 112)
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)

    # Concatenate along the first dimension to get (2, 3, 112, 112)
    concatenated_tensor = torch.cat((tensor1, tensor2), dim=0)

    return concatenated_tensor

import io

from tqdm import tqdm
def main(args): 
    target=args.target  

    file_path ="./Dataset_Test/"
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize


    #filename = "Original_ViT"+target+".bin"
    
    filename = "/GViT"+target+".bin"
  
    """
    #SAVE test dataset so i can use it as a dataloader
    TARGET = [i for i in target.split(',')]
    ver = get_val_data('./eval/', TARGET)
    issame=ver[0][2]
    test_dataset=ver[0][1][0]
    print(test_dataset.shape)

    

     # Specify the path to the directory where the file will be saved
    filename = "/GViT"+target+".bin"  # Specify the name of the binary file

     
    with open(file_path + filename, "wb") as fout:
        for i in range(len(issame)):
            boolean_bytes = int(issame[i]).to_bytes(1, byteorder='little')

            output1 = test_dataset[i * 2]
            output2 = test_dataset[i * 2 + 1]
            
            # Convert output1 and output2 to numpy arrays
            output1_np = np.array(output1)
            output2_np = np.array(output2)
            
            # Convert output1 and output2 to bytes
            output1_bytes = output1_np.astype(dtype).tobytes()
            output2_bytes = output2_np.astype(dtype).tobytes()
            
            # Write the bytes to the binary file
            
            # Write boolean_bytes, output_bytes
            fout.write(boolean_bytes)
            fout.write(output1_bytes)
            fout.write(output2_bytes)  
    """
    

    file_size = os.path.getsize(file_path + filename)
    # Calculate the total size occupied by one pair
    pair_size = 1 + 3 * 112 * 112 * itemsize*2

    # Calculate the number of pairs
    Original_pair_count = file_size // pair_size

    print(Original_pair_count)





    Original_dataset=TestDataset_Original(file_path +"GViT"+target+".bin",0,Original_pair_count,batch_size=20)
    issame=[]
    
    
    for i in range(Original_pair_count):
      issame.append(Original_dataset.boolean(i))
    
    
    

    print("issame list")   
    
    model=VisionTransformer(
                         num_classes = 93431,
                         img_size=112,
                         patch_size=8,
                         embed_dim=512,
                         depth=20,
                         num_heads=8,
                         drop_rate=0.1,
                     ).cuda()

    
    model.load_state_dict(torch.load("results/ViT_LRP_1a_treinar/Backbone_VIT_LRP_Epoch_27_Batch_13962000_Time_2023-03-13-13-23_checkpoint.pth"))

    MULTI_GPU = False
    DEVICE = torch.device("cuda:0")
    accuracy, std, _xnorm, best_threshold, roc_curve_tensor,dist,_,tpr,fpr=perform_val_without_flip_dataloader(MULTI_GPU, DEVICE, 512, model, Original_dataset, issame)
  


    print(accuracy)
    print(best_threshold)
   

    tpr, fpr, acc,tnr,fnr=calculate_accuracy(best_threshold, dist, issame)
    print(tpr)
    print(tnr)
  
    
    predict_issame = np.less(dist, best_threshold)
    
    tp = (np.logical_and(predict_issame, list)) #true positives
    fp = (np.logical_and(predict_issame, np.logical_not(issame))) #false positives
    tn = (np.logical_and(np.logical_not(predict_issame), np.logical_not(issame))) #true negatives
    fn = (np.logical_and(np.logical_not(predict_issame), issame)) #false negatives


    
    indices=[np.where(tp)[0][0],np.where(tp)[0][1],np.where(tp)[0][2],np.where(tp)[0][3],
             np.where(tp)[0][4],np.where(tp)[0][5],np.where(tp)[0][6],np.where(tp)[0][7]]

    attribution_generator = LRP(model) 


    fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(10, 10))
    for i in range(4):
        # Plot image in the first column
  

        #axs[i, 0].imshow(tensor_to_image(torch.zeros(3, 112, 112).permute((1,2,0))))
        axs[i, 0].imshow(tensor_to_image(Original_dataset[indices[i]*2+1].permute((1,2,0))))
        axs[i, 0].set_axis_off()

        axs[i, 1].imshow(tensor_to_image(Original_dataset[indices[i]*2].permute((1,2,0))))
        axs[i, 1].set_axis_off()


      
        image_transformer_attribution = Original_dataset[indices[i]*2].permute(1, 2, 0).data.cpu().numpy()
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())


        Intersection_heatmap,_,_,_,_,_,_,_,_,_=Explainability_heatmap(attribution_generator,Original_dataset.read_pair_outputs(indices[i]),method="Attention_Relevance_Scores")

        vis = show_cam_on_image(image_transformer_attribution,Intersection_heatmap)
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        axs[i, 2].imshow(vis)
        if i==0:
            axs[i, 2].set_title("No-Grad ViT-LRP \n Intersection",fontdict={'fontsize': 8})
        axs[i, 2].set_axis_off()




        Intersection_heatmap,_,_,_,_,_, _,_,_,_=Explainability_heatmap(attribution_generator,Original_dataset.read_pair_outputs(indices[i]),method="transformer_attribution")



        vis = show_cam_on_image(image_transformer_attribution,Intersection_heatmap)
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        axs[i, 3].imshow(vis)
        if i==0:
            axs[i, 3].set_title("No-Grad ViT-LRP \n Intersection",fontdict={'fontsize': 8})
        axs[i, 3].set_axis_off()



        Intersection_heatmap,_,_,_,_,_, _,_,_,_=Explainability_heatmap(attribution_generator,Original_dataset.read_pair_outputs(indices[i]),method="Grad_Rollout")



        vis = show_cam_on_image(image_transformer_attribution,Intersection_heatmap)
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        axs[i, 4].imshow(vis)
        if i==0:
            axs[i, 4].set_title("Grad-Rollout \n Intersection",fontdict={'fontsize': 8})
        axs[i, 4].set_axis_off()



        Intersection_heatmap,_,_,_,_,_, _,_,_,_=Explainability_heatmap(attribution_generator,Original_dataset.read_pair_outputs(indices[i]),method="rollout")



        vis = show_cam_on_image(image_transformer_attribution,Intersection_heatmap)
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        axs[i, 5].imshow(vis)
        if i==0:
            axs[i, 5].set_title("Rollout \n Intersection",fontdict={'fontsize': 8})
        axs[i, 5].set_axis_off()


        
        Ymp,Smp=MinPlus_Original(Original_dataset[indices[i]*2],Original_dataset[indices[i]*2+1],model)
        axs[i, 6].imshow(cv2.cvtColor(Ymp, cv2.COLOR_BGR2RGB))
        axs[i, 6].set_axis_off()
        if i==0:
            axs[i, 6].set_title(" MinPlus",fontdict={'fontsize': 8})


        Ylime=LIME_Original(Original_dataset[indices[i]*2],Original_dataset[indices[i]*2+1],model)
        axs[i, 7].imshow(cv2.cvtColor(Ylime, cv2.COLOR_BGR2RGB))
        axs[i, 7].set_axis_off()
        if i==0:
            axs[i, 7].set_title("LIME",fontdict={'fontsize': 8})
        

    fig.savefig("figure1.png")
    
    
    
   

    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help='', default="lfw")
    
    return parser.parse_args(argv)

import sys
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:])) 