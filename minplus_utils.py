import numpy as np
import cv2
import matplotlib.pyplot as plt
from   tqdm import tqdm
import time
import random
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from   sklearn.linear_model import LinearRegression

# MISCELANEOUS
def num2fixstr(x,d):
    st = '%0*d' % (d,x)
    return st
def str2(x):
    return "{:.2f}".format(x)
def str4(x):
    return "{:.4f}".format(x)

# INPUT OUTPUT FUNCTIONS
def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256,256)) 
    return img

def imshow(I,height=6,width=None,show_pause=0,title=None,fpath=None):
  n = height
  if width!=None:
    m = width
  else:
    N = I.shape[0]
    M = I.shape[1]
    m = round(n*M/N)
  __,ax = plt.subplots(1,1,figsize=(m,n))
  bw = len(I.shape)==2
  if bw:
     ax.imshow(I)
  else:
     ax.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
  if title!=None:
    ax.set_title(title)
  plt.axis('off')
  if fpath!=None:
     plt.savefig (fpath,bbox_inches = 'tight',pad_inches = 0)
  if show_pause>0:
    plt.pause(show_pause)
    plt.close()
  else:
    plt.show()


def contours(D,A,color_map,contour_levels,print_levels=False,color_levels=True):
    # D: heatmap
    # A: background image
    # Examples
    # contours(D,A,'jet'  ,10,print_levels=False,color_levels=True,img_file='Cont.png')
    # contours(D,A,'white',10,print_levels=True,color_levels=False,img_file=None)
    height    = D.shape[0]
    width     = D.shape[1]
    levels    = np.linspace(0.1, 1.0, contour_levels)
    x         = np.arange(0, width, 1)
    y         = np.arange(0, height, 1)
    extent    = (x.min(), x.max(), y.min(), y.max())

    Z = D/D.max()
    At = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(At,extent=extent)
    if color_levels:
      CS = plt.contour(Z, levels, cmap=color_map, origin='upper', extent=extent)
    else:
      CS = plt.contour(Z, levels, colors=color_map, origin='upper', extent=extent)

    if print_levels:
      plt.clabel(CS,fontsize=9, inline=1)
    plt.axis('off')
    plt.savefig ('Contour.png',bbox_inches = 'tight',pad_inches = 0)
    # plt.show()
    plt.clf()
    plt.close()
    C = cv2.imread('Contour.png')
    C = cv2.resize(C,(width,height))
    return C


## IMAGE PROCESSING
def gaussian_kernel(size, sigma, type='Sum'):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1,
           -size // 2 + 1:size // 2 + 1]
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    if type=='Sum':
      kernel = kernel / kernel.sum()
    else:
      kernel = kernel / kernel.max()
    return kernel.astype('double')

def DefineGaussianMask(ic,jc,nh,N=112,M=112):
  # Define an image of NxM, with a Gaussian of nh x nh centred in (ic,jc)
  nh2 = round(nh/2)
  i1  = ic
  j1  = jc
  n   = N+nh
  m   = M+nh
  s   = nh/8.5
  h   = 1-gaussian_kernel(nh,s,type='Max')
  Mk  = np.ones((n,m))
  i2  = i1+nh
  j2  = j1+nh
  Mk[i1:i2,j1:j2] = h
  return Mk[nh2:nh2+N,nh2:nh2+M]

def MaskMult(A,Mk):
    n = Mk.shape[0]
    m = Mk.shape[1]
    M = np.zeros((n,m,3))
    M[:,:,0] = Mk
    M[:,:,1] = Mk
    M[:,:,2] = Mk
    Ak = np.multiply(M,A)
    return Ak.astype(np.float32)
    return Ak.astype(np.uint8)

def MaskMultNeg(BB,Bo,Mk):
    B1 = MaskMult(Bo,1-Mk)
    Bn = (255-B1)/255
    Bk = 255 - np.multiply(BB,Bn)
    return Bk.astype(np.uint8)

def heatmap(Ao,H,ss,alpha=0.5,type='Max',count1=np.ones((112,112))):
    hm = gaussian_kernel(ss,ss/8.5,type='Max')
    X  = cv2.filter2D(H,-1,hm)
    D = X-np.min(X)
    if type=='Max':
        #D = minmax_norm(X)
        D = D/np.max(D)
    else:
        D = D/np.sum(D)
    X  = np.uint8(D*255)
    HM = cv2.applyColorMap(X, cv2.COLORMAP_JET)
    Y  = cv2.addWeighted(HM, alpha, Ao, 1-alpha, 0)
    return D,Y

# MINUS FUNCTIONS
def removing_score(A,fx_score,d,nh,ROI=None,background=1.0):
  N      = A.shape[0]
  M      = A.shape[1]
  Hsc    = background*np.ones((N,M))
  sc_min = 10
  sc_max = -10
  if ROI is None:
      ROI = np.ones((N,M,3))
  for ic in range(d,N,d):
    for jc in range(d,M,d):
      if np.sum(ROI[ic,jc,:])>0:
        Mk  = DefineGaussianMask(ic,jc,nh)
        Ak  = MaskMult(A,Mk)
        #print(Ak.shape)
        sc = fx_score(Ak)
        Hsc[ic,jc] = sc
        if sc<sc_min:
            sc_min  = sc
            out_min = [Ak,sc_min,ic,jc]
        if sc>sc_max:
            sc_max  = sc
            out_max = [Ak,sc_max,ic,jc]
  return Hsc,out_min,out_max

def saliency_minus(A,fx_score,nh,d,n,nmod,th):
    N    = A.shape[0]
    M    = A.shape[1]  
 
    H1   = np.zeros((N,M))
 
    sc0 = fx_score(A)
   
    #st   = "{:7.4f}".format(sc0)
    #print('minus t = 000 sc[0]='+st)
    #imshow(A,show_pause=1,title='sc0 = '+st)
    t    = 0
    sct  = sc0
    #print(sc0)
    At   = A
    dsc  = 1
    t0   = time.time()
    if sct<th:
       return np.zeros((N,M)),np.zeros((N,M))
    while sct>th and t<n: 
        t = t+1
        Hsc,out_min,__ = removing_score(At,fx_score,d,nh,ROI=At,background=sc0)
        if t==1:
            H0 = sc0-Hsc
        sct     = out_min[1]       # minimal score by removing a gaussian mask centered in (i,j)
        i       = out_min[2]
        j       = out_min[3]
        #sct_st  = "{:7.4f}".format(sct)
        dsc     = sc0-sct
        H1[i,j] = dsc
        sc0 = sct
        #hij_st = "{:.4f}".format(H1[i,j])
        t1 = time.time()  
        #dt_st = "{:.2f}".format(t1-t0)
        t0     = t1  
        #print('minus t = '+num2fixstr(t,3)+' sc[t]='+sct_st+ ' H[' + num2fixstr(i,3)+ ',' +num2fixstr(j,3) + '] = sc[t-1]-sc[t] = '+hij_st+' > '+dt_st+'s')
        At      = out_min[0]
        """
        if np.mod(t,nmod)==0:
            imshow(At,show_pause=1)
        """
    return H0,H1

# PLUS FUNCTIONS
def adding_score(A,Ao,fx_score,d,nh,ROI=None,background=1.0):

  N      = A.shape[0]
  M      = A.shape[1]
  nh2    = round(nh/2)
  sc_max = -10
  sc_min = 10
  AA     = 255-A
  Hsc    = np.zeros((N,M))
  out_min = None
  out_max = None
  if ROI is None:
      ROI = np.ones((N,M,3))
  for ic in range(d,N,d):
    for jc in range(d,M,d):
      if np.sum(ROI[ic,jc,:])>0:
        Mk = DefineGaussianMask(ic,jc,nh)
        Ak = MaskMultNeg(AA,Ao,Mk)
        sc  = fx_score(Ak)
        Hsc[ic,jc] = sc
        if sc<sc_min:
           sc_min  = sc
           out_min = [Ak,sc_min,ic,jc]
        if sc>sc_max:
           sc_max  = sc
           out_max = [Ak,sc_max,ic,jc]
  return Hsc,out_min,out_max

def add_best(A,Ao,fx_score,d,nh):
  Hsc,__,out_max = adding_score(A,Ao,fx_score,d,nh)
  Aks = out_max[0]
  output = out_max[1:]
  return Aks,Hsc,output

def saliency_plus(A,fx_score,nh,d,n,nmod,th):
    N    = A.shape[0]
    M    = A.shape[1]  
    At   = np.random.rand(N,M,3)*2
    At   = At.astype(np.uint8)
    H1   = np.zeros((N,M))
  
    
    scA  = fx_score(A)
    
    sc0  = fx_score(At)
    #print("scA {} sc0 {}".format(scA,sc0))
    #st   = "{:7.4f}".format(sc0)
    #print('plus  t = 000 sc[0]='+st)
    #imshow(At,show_pause=1)
    sct  = sc0
    t    = 0
    dsc  = 1
    t0   = time.time()
    if (scA-sct) < th:
       return np.zeros((N,M)),np.zeros((N,M))
    while t<n and (scA-sct)>th:
        t = t+1
        At,Hsc,out_max = add_best(At,A,fx_score,d,nh)
        sct = out_max[0]
        if t==1:
            H0 = Hsc-sc0
        i       = out_max[1]
        j       = out_max[2]
        #sct_st  = "{:7.4f}".format(sct)
        dsc     = sct-sc0
        H1[i,j] = dsc
        sc0     = sct
        #hij_st  = "{:.4f}".format(H1[i,j])
        t1 = time.time()  
        #dt_st = "{:.2f}".format(t1-t0)
        t0     = t1  
        #print('plus  t = '+num2fixstr(t,3)+' sc[t]='+sct_st+ ' H[' + num2fixstr(i,3)+ ',' +num2fixstr(j,3) + '] = sc[t]-sc[t-1] = '+hij_st+' > '+dt_st+'s')
        """
        if np.mod(t,nmod)==0:
            imshow(At,show_pause=1)
        """
    return H0,H1

def randomints(i1,i2,n):
  randomlist = []
  i2 = i2-1
  for i in range(n):
    x = random.randint(i1,i2)
    randomlist.append(x)
  return randomlist

def DefineRISEmasks(k,N,M,smin,smax,kernel='Gauss'):
  height = 2*N
  width  = 2*M
  ii = randomints(0, height, k)
  jj = randomints(0, width, k)
  ss = randomints(smin,smax,k)
  Mk  = np.ones((height,width))
  for t in range(len(ii)):
      i1 = ii[t]
      j1 = jj[t]
      s = 1.0*ss[t]
      if kernel=='Gauss':
         h  = 1-gaussian_kernel(s,s/8.5,type='Max')
      else:
         h = 1.0*np.zeros((ss[t],ss[t]))
      n = h.shape[0]
      i2 = i1+n
      j2 = j1+n
      if i2>height:
        i2 = height
      if j2>width:
        j2 = width
      Mk[i1:i2,j1:j2] = np.multiply(Mk[i1:i2,j1:j2],h[0:i2-i1,0:j2-j1])

  i1 = round(N/2)
  j1 = round(M/2)
  if kernel=='Square':
    Mks = cv2.resize(Mk,(32,32))
    Mk  = cv2.resize(Mks,(height,width))
  return Mk[i1:i1+N,j1:j1+M]

def saliency_RISE(A,fx_score,n,k,smin,smax,kernel='Gauss'):
    N      = A.shape[0]
    M      = A.shape[1]
    #xA     = fattribute(A)
    S      = np.zeros((N,M))
    sc_max = 10
    for t in range(n):
      Mk  = 1-DefineRISEmasks(k,N,M,smin,smax,kernel=kernel)
      At  = MaskMult(A,Mk)
      #xAt = fattribute(At)
      #sc  = xAt[kexp]
      sc  = fx_score(At)
      S   = S+sc*Mk
      if sc<sc_max:
        sc_max = sc
    #print('sc_min=',sc)
    S = S/n
    return S


def perturbations(n,m,p=0.5):
    perts = np.random.binomial(1, p, size=(n,m))
    return perts

def perturb_image(img,perturbation,segments):
    active_pixels = np.where(perturbation == 1)[0] #superpixels that are active
    #segments has values between 0 and 64
    mask = np.zeros(segments.shape) #112,112
    #print(segments)
    for active in active_pixels:
        mask[segments == active] = 1 #part of the image left untouched
    
    perturbed_image = copy.deepcopy(img)
    X = perturbed_image*mask[:,:,np.newaxis]
    X = X.astype(np.uint8)
    #imshow(X,fpath='Image_LIME.png')
    return X,mask


def perturb_image_score_to_masked_Areas(img,perturbation,segments):
    active_pixels = np.where(perturbation == 1)[0] #superpixels that are active
    #segments has values between 0 and 64
    mask = np.zeros(segments.shape) #112,112
    #print(segments)
    for active in active_pixels:
        mask[segments == active] = 1 #part of the image left untouched
    
    perturbed_image = copy.deepcopy(img)
    X = perturbed_image*mask[:,:,np.newaxis]
    X = X.astype(np.uint8)
    #imshow(X,fpath='Image_LIME.png')
    return X,1-mask

def color_mask(X,M,col):
    Y = copy.deepcopy(X)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]==0:
                Y[i,j,:]=col
    return Y

def show_superpixels(B,superpixels):
    Bs = skimage.segmentation.mark_boundaries(B, superpixels)
    Bs = Bs*255
    Bs = Bs.astype(np.uint8)
    return Bs


def perturb_image_new(img,perturbation,segments,count1):
    active_pixels = np.where(perturbation == 1)[0] #superpixels that are active
    #segments has values between 0 and 64
    mask = np.zeros(segments.shape) #112,112
    #print(segments)
    for active in active_pixels:
        mask[segments == active] = 1 #part of the image left untouched
    mask[count1==0]=0
    perturbed_image = copy.deepcopy(img)
    X = perturbed_image*mask[:,:,np.newaxis]
    X = X.astype(np.uint8)
    #imshow(X,fpath='Image_LIME.png')
    return X,mask


def saliency_LIME_Multiple_ViTs(A,fx_score,N=500,num_top_features=16,kernel_size=3,max_dist=200, ratio=0.2,count1=np.ones((112,112))):
    #xA          = fattribute(A)
    #sc          = xA[kexp]
    sc          = fx_score(A)
    #print('score = '+str4(sc))
    superpixels = skimage.segmentation.quickshift( A , kernel_size=kernel_size,max_dist=max_dist, ratio=ratio)
    
    m           = np.unique(superpixels).shape[0]
    
    perts       = perturbations(N,m)
    
    Y = A
    #imshow(Y)
    scores = []
    s = np.zeros((A.shape[0],A.shape[1]))
    for i in range(N):
        At,Mt = perturb_image_score_to_masked_Areas(A,perts[i],superpixels)
        
        #xA    = fattribute(At)
        #sc    = xA[kexp]
        sc    = fx_score(At)
        s     = s + sc*Mt*count1
        scores.append(sc)
    s = s/N
    predictions = np.array(scores)
    original_image = np.ones(m)[np.newaxis,:] #Perturbation with all superpixels enabled 
    distances      = sklearn.metrics.pairwise_distances(perts,original_image, metric='cosine').ravel()
    kernel_width   = 0.25
    weights        = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
    simpler_model  = LinearRegression()
    simpler_model.fit(X=perts, y=predictions, sample_weight=weights)
    coeff          = simpler_model.coef_
    ys             = simpler_model.predict(perts)
    err            = np.abs(ys-predictions)
    #print('error = '+str4(np.mean(err))) 
    top_features   = np.argsort(coeff)[-num_top_features:] 
    mask = np.ones(m) 
    mask[top_features]= False #Deactivatetop superpixels
    As,Ms = perturb_image_score_to_masked_Areas(A,mask,superpixels)
   
    """
    mask = np.zeros(m) 
    mask[top_features]= True #Activate top superpixels
    As,Ms = perturb_image_score_to_masked_Areas(A,mask,superpixels)
    """
    return As,Ms,s,superpixels


def saliency_LIME_masked_areas_score(A,fx_score,N=500,num_top_features=16,kernel_size=3,max_dist=200, ratio=0.2):
    #xA          = fattribute(A)
    #sc          = xA[kexp]
    sc          = fx_score(A)
    #print('score = '+str4(sc))
    superpixels = skimage.segmentation.quickshift( A , kernel_size=kernel_size,max_dist=max_dist, ratio=ratio)
    
    m           = np.unique(superpixels).shape[0]
    
    perts       = perturbations(N,m)
    
    Y = A
    #imshow(Y)
    scores = []
    s = np.zeros((A.shape[0],A.shape[1]))
    for i in range(N):
        At,Mt = perturb_image_score_to_masked_Areas(A,perts[i],superpixels)
        #imshow(At,fpath='wtf')
        #xA    = fattribute(At)
        #sc    = xA[kexp]
        sc    = fx_score(At)
        s     = s + sc*Mt
        scores.append(sc)
    s = s/N
    predictions = np.array(scores) #array of scores predictions
    original_image = np.ones(m)[np.newaxis,:] #Perturbation with all superpixels enabled 
    distances      = sklearn.metrics.pairwise_distances(perts,original_image, metric='cosine').ravel()
    kernel_width   = 0.25
    weights        = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
    simpler_model  = LinearRegression()
    simpler_model.fit(X=perts, y=predictions, sample_weight=weights)
    coeff          = simpler_model.coef_
    ys             = simpler_model.predict(perts)
    err            = np.abs(ys-predictions)
    #print('error = '+str4(np.mean(err))) 
    top_features   = np.argsort(coeff)
    print(coeff.shape)
    """ 
    mask = np.ones(m) 
    mask[top_features]= 0 #Deactivatetop superpixels
    As,Ms = perturb_image_score_to_masked_Areas(A,mask,superpixels)
    """
    print(top_features[0,-16:])
    mask = np.ones(m) 
    mask[top_features[0,-num_top_features:]]= False #Activate top superpixels
    As,Ms = perturb_image_score_to_masked_Areas(A,mask,superpixels)
    #imshow(As,fpath='wtf')
    
    return As,Ms,s,superpixels

def saliency_LIME(A,fx_score,N=500,num_top_features=16,kernel_size=3,max_dist=200, ratio=0.2):
    #xA          = fattribute(A)
    #sc          = xA[kexp]
    sc          = fx_score(A)
    #print('score = '+str4(sc))
    superpixels = skimage.segmentation.quickshift( A , kernel_size=kernel_size,max_dist=max_dist, ratio=ratio)
    
    m           = np.unique(superpixels).shape[0]
    
    perts       = perturbations(N,m)
    
    Y = A
    #imshow(Y)
    scores = []
    s = np.zeros((A.shape[0],A.shape[1]))
    for i in range(N):
        At,Mt = perturb_image(A,perts[i],superpixels)
        #xA    = fattribute(At)
        #sc    = xA[kexp]
        sc    = fx_score(At)
        s     = s + sc*Mt
        scores.append(sc)
    s = s/N
    predictions = np.array(scores)
    original_image = np.ones(m)[np.newaxis,:] #Perturbation with all superpixels enabled 
    distances      = sklearn.metrics.pairwise_distances(perts,original_image, metric='cosine').ravel()
    kernel_width   = 0.25
    weights        = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
    simpler_model  = LinearRegression()
    simpler_model.fit(X=perts, y=predictions, sample_weight=weights)
    coeff          = simpler_model.coef_
    ys             = simpler_model.predict(perts)
    err            = np.abs(ys-predictions)
    #print('error = '+str4(np.mean(err))) 
    top_features   = np.argsort(coeff)[-num_top_features:] 
    #print(top_features)
    mask = np.zeros(m) 
    mask[top_features]= 1 #Activate top superpixels
    As,Ms = perturb_image(A,mask,superpixels)
    #imshow(As,fpath='wtf')
    return As,Ms,s,superpixels

