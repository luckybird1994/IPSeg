import os
import torch
torch.set_num_threads(16)
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from extractor_dino import ViTExtractor
from extractor_sd import load_model, process_features_and_mask
import torch.nn.functional as F

def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

def compute_pck(model, aug, files,  real_size=960):
    # load DINO v2
    MODEL_SIZE = args.MODEL_SIZE
    img_size = 840 
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
    model_type = model_dict[MODEL_SIZE] 
    layer = 11 
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' 
    stride = 14
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)


    N = len(files) 
    pbar = tqdm(total=N)

    for idx in range(N):
        # load image 
        img1 = Image.open(files[idx]).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=False)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=False)

        # load text prompot 
        if args.TEXT_INPUT: input_text = open(files[idx].replace('img','blip2-caption')[:-4]+'.txt').read().strip()
        else: input_text = ""
        
        # extract feature
        with torch.no_grad():
            img1_desc = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False, raw=True)
            img1_batch = extractor.preprocess_pil(img1)
            img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
            
            state_dict = {}
            state_dict['sd_feat'] = img1_desc
            state_dict['dino_feat'] = img1_desc_dino            
            torch.save(state_dict,files[idx].replace('imgs','sd_raw+dino_feat')[:-4]+'.pth')

def main(args):
    # random setting
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.benchmark = True
    
    # load stable diffusion
    model, aug = load_model(diffusion_ver=args.VER, image_size=args.SIZE, num_timesteps=args.TIMESTEP, block_indices=tuple(args.INDICES))

    # prepare path
    imgs_dir = os.path.join(args.path, 'imgs')
    global feat_save_dir, captions_dir
    captions_dir = os.path.join(args.path, 'blip2_caption')
    feat_save_dir = os.path.join(args.path, 'sd_raw+dino_feat')
    if not os.path.exists(feat_save_dir):
        os.makedirs(feat_save_dir)
        
    # feature extract loop    
    categories = os.listdir(imgs_dir)
    categories = sorted(categories)
    for cat in categories:
        if cat.startswith('.'): continue
        if not os.path.exists(feat_save_dir + '/' +cat):
            os.makedirs(feat_save_dir + '/' +cat)
        
        files = os.listdir(os.path.join(imgs_dir,cat))
        files = sorted(files)
        files = [os.path.join(imgs_dir,cat,file) for file in files]
        compute_pck(model, aug, feat_save_dir + '/' +cat,files)

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=7 python extract_sd_raw+dino.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/data/tanglv/data/fss-te/fold0')
    parser.add_argument('--SEED', type=int, default=42)

    # Stable Diffusion Setting
    parser.add_argument('--VER', type=str, default="v1-5")                          # version of diffusion, v1-3, v1-4, v1-5, v2-1-base
    parser.add_argument('--PROJ_LAYER', action='store_true', default=False)         # set true to use the pretrained projection layer from ODISE for dimension reduction
    parser.add_argument('--SIZE', type=int, default=960)                            # image size for the sd input
    parser.add_argument('--INDICES', nargs=4, type=int, default=[2,5,8,11])         # select different layers of sd features, only the first three are used by default
    parser.add_argument('--WEIGHT', nargs=5, type=float, default=[1,1,1,1,1])       # first three corresponde to three layers for the sd features, and the last two for the ensembled sd/dino features
    parser.add_argument('--TIMESTEP', type=int, default=100)                        # timestep for diffusion, [0, 1000], 0 for no noise added

    # DINO Setting
    parser.add_argument('--MODEL_SIZE', type=str, default='base')                   # model size of thye dinov2, small, base, large
    parser.add_argument('--TEXT_INPUT', action='store_true', default=False)         # set true to use the explicit text input
    parser.add_argument('--NOTE', type=str, default='')

    args = parser.parse_args()
    main(args)