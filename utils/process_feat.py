import torch
from torch.nn import functional as F
from PIL import Image
 

def clip_feat(feature, img_path):
    # feature1 shape (1,1,3600,768*2)
    feature = feature.squeeze()
    chennel_dim = feature.shape[-1]
    
    num_patches = int((feature.shape[-2])**(0.5))
    
    #print(num_patches)
    h, w = Image.open(img_path).size
    scale_h = h/num_patches
    scale_w = w/num_patches
     
    if scale_h > scale_w:
        scale = scale_h
        scaled_w = int(w/scale)
        feature = feature.reshape(num_patches,num_patches,chennel_dim)
        feature_uncropped=feature[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
    elif scale_h < scale_w:
        scale = scale_w
        scaled_h = int(h/scale)
        feature = feature.reshape(num_patches,num_patches,chennel_dim)
        feature_uncropped=feature[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]
    else:
        feature_uncropped=feature.reshape(num_patches,num_patches,chennel_dim)
         
    return feature_uncropped #[H W C]