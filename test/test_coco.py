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
device = torch.device("cuda:1")  
# 定义图像和深度图的变换
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
# depth_model.to(device)
depth_model.to(torch.float32) 


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None,depth_transform=None):
        self.coco = CocoCaptions(root=root, annFile=annFile, transform=None)
        self.transform = transform
        self.depth_transform = depth_transform
        self.image_paths=[]
        for i in range(len(self.coco)):
            img_info = self.coco.coco.loadImgs(self.coco.ids[i])[0]
            img_path = os.path.join(self.coco.root, img_info['file_name'])
            self.image_paths.append(img_path)
        

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        image, captions = self.coco[idx]
        img_path = self.image_paths[idx]
        if self.transform:
            image = self.transform(image).to(device) 
            depth = depth_model(image.unsqueeze(0))
            depth_map = depth.squeeze(0)

        captions = captions[:5]  
        captions = clip.tokenize(captions)
        return image, depth_map,captions

def test_epoch_retrieval(model):
        image_features = []
        text_features = []
        coco_dataset = CocoDataset(root="/home/maiyubo/llm/data/coco/val2017", annFile="/home/maiyubo/llm/data/coco/captions_val2017.json", transform=clip_transform,depth_transform=depth_transform)
        dataloader = torch.utils.data.DataLoader(coco_dataset, batch_size=256, num_workers=0, pin_memory=False)

        with torch.no_grad():
            for images, depth,captions_batch in dataloader:
                images = images.to(device)
                depth = depth.to(device)
                # import pdb;pdb.set_trace()
                image_features.append(model.visual(images,depth,pos_embed=""))
                for captions in captions_batch:
                    caption_input=captions.to(device)
                    text_features.append(model.encode_text(caption_input))
            
            image_features = torch.cat(image_features) #[5000,512]
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = torch.stack(text_features) #[5000,5,512]
            text_features = text_features.view(-1,text_features.size(-1))
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.T
            # I2T (Image to Text) 
            i2t_accuracies = []
            for k in [1, 5, 10]:
                pred_true = 0
                for i in range(5000):
                    pred = similarity[i]
                    b = pred.argsort()[-k:]
                    for j in range(5):
                        true_index = 5 * i + j
                        if true_index in b:
                            pred_true += 1
                            break
                i2t_accuracies.append(pred_true / 5000)
                print("acc",pred_true / 5000)

            # T2I (Text to Image) 
            t2i_accuracies = []
            similarity = similarity.T
            for k in [1, 5, 10]:
                pred_true = 0
                for i in range(25000):
                    pred = similarity[i]
                    b = pred.argsort()[-k:]
                    true_index = i // 5
                    if true_index in b:
                        pred_true += 1
                t2i_accuracies.append(pred_true / 25000)
                print("acc",pred_true / 25000)

        return i2t_accuracies, t2i_accuracies


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
    i2t_acc, t2i_acc = test_epoch_retrieval(model)
    print("I2T Accuracies:", i2t_acc)
    print("T2I Accuracies:", t2i_acc)
    


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
