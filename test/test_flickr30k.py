import os
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append('../Depth-Anything-V2')
import spat_clip as sclip
import clip
from collections import defaultdict
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from PIL import Image
import argparse
import json
import torch
device = torch.device("cuda:0")  
depth_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
])

def to_rgb(image):
    return image.convert("RGB")

clip_transform = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(336),
    to_rgb,  
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
encoder = 'vitl'

depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(device).eval() 
depth_model.to(torch.float32) 


    
def get_accuracy_i2t(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        for i in range(image_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            for j in range(5):
                true_index = 5*i + j
                if true_index in topk:
                    pred_true = pred_true + 1
                    break

        print(pred_true/image_feature.shape[0])


def get_accuracy_t2i(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (text_feature @ image_feature.T).softmax(dim=-1)

        for i in range(text_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            true_index = i//5
            if true_index in topk:
                pred_true = pred_true + 1

        print(pred_true/text_feature.shape[0])



def zeroshot_classifier_ours(captions, model, local_rank=0):
    with torch.no_grad():
        zeroshot_weights = []
        for caption in tqdm(captions):
            texts = sclip.tokenize(caption).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
    return zeroshot_weights

@torch.no_grad()
def test_flickr(model):
    text_list = []
    feature_list = []
    with torch.no_grad():
        with open("/home/maiyubo/llm/data/results_20130124.token", 'r') as f:
            dataset = f.readlines()
            for data in dataset:
                image = data.split('\t')[0]
                text = data.split('\t')[1]
                text_list.append(text)
        len_list = len(text_list)
        print(len_list)
        for i in range(20):
            text = text_list[i*len_list//20: (i+1)*len_list//20]
            text = clip.tokenize(text, truncate=True).to(device)
            feature_list.append(model.encode_text(text))
        text_feature = torch.concatenate(feature_list, dim=0)

        # data_root = "/home/maiyubo/llm/data/f30k/flickr30k-images"
        data_root = "../dataset/flickr30k/flickr30k-images"
        print("begin image")
        img_feature_list = []
        with open("../dataset/flickr30k/results_20130124.token", 'r') as f:
            dataset = f.readlines()
            data_len = len(dataset)
            for i in range(data_len//5):
                data = dataset[5*i]
                image_name = data.split('\t')[0][:-2]
                image = Image.open(os.path.join(data_root,image_name)).convert('RGB')
                image = clip_transform(image).unsqueeze(0).to(device)
                depth =depth_model(image)
                img_feature = model.visual(image,depth,pos_embed="")
                img_feature_list.append(img_feature)
                torch.cuda.empty_cache()
                del img_feature, image
        image_feature = torch.concatenate(img_feature_list, dim=0)
    print("begin process")
    get_accuracy_i2t(text_feature, image_feature, 1)
    get_accuracy_i2t(text_feature, image_feature, 5)
    # get_accuracy_i2t(text_feature, image_feature, 10)
    get_accuracy_t2i(text_feature, image_feature, 1)
    get_accuracy_t2i(text_feature, image_feature, 5)
    # get_accuracy_t2i(text_feature, image_feature, 10)
    return None

def main(args):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = sclip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)
    resume_pth = '../checkpoint/iter_5000.pth'
    ckpt = torch.load(resume_pth, map_location="cpu")
    new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    info=model.load_state_dict(new_ckpt, strict=True)
    print(info)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  
    model.to(device)
    test_flickr(model)
    

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CLIP inference')
    args.add_argument('-d', '--data-dir', default='../data/imagenet-val', type=str,
                      help='dataset path (default: None)')
    args.add_argument('-w', '--num-workers', default=8, type=int,
                      help='number of workers (default: 64)')
    args.add_argument('-b', '--batch_size', default=1, type=int,
                      help='Batch size (default: 64)')

    config = args.parse_args()
    main(config)





