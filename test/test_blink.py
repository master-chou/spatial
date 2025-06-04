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
depth_model = depth_model.eval() 
depth_model.to(torch.float32) 


class BLINK:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.task = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['idx']] = data['captions']
                    self.imgs[data['idx']] = data['image_1']
                    self.answers[data['idx']] = data['answer']
                    self.task[data['idx']] = data['sub_task']

class BLINK_Benchmark(Dataset):
    def __init__(self,tasks):
        self.root_dir = '../dataset/BLINK/Spatial_Relation'
        print(self.root_dir)
        self.dataset = BLINK(os.path.join(self.root_dir,'test.json'))
        self.image_ids = list(self.dataset.imgs.keys())
        self.captions = [x for x in self.dataset.anns.values()]
        self.transform = clip_transform
        self.depth_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
        ])
        self.depth_model = depth_model

      
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'output_images',self.dataset.imgs[img_ids])
        image = Image.open(image_path).convert('RGB')        
        answer = self.dataset.answers[img_ids]
        if self.transform:
            image = self.transform(image)
            depth = self.depth_model(image.unsqueeze(0))
            depth = depth.squeeze(0)
        return image, depth, answer



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

@torch.no_grad()
def test_epoch_blink(model,dataloader,text_embeddings):
    corr_pred = 0
    total_num = 0
    for i, (images, depth, answer) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        answer = answer.to(device)
        depth = depth.to(device)
        image_features = model.visual(images, depth,pos_embed="")
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = image_features.unsqueeze(1).repeat([1, text_embeddings.size(-2), 1]) * text_embeddings
        pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
        corr_pred += torch.eq(pred, answer).sum()
        total_num += images.size(0)
    return corr_pred / total_num

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
def test_blink(model):
    totalacc=0
    count = 0
    model.visual.eval()
    testset = BLINK_Benchmark('none_spatial.json')
    text_embeddings = zeroshot_classifier_ours(testset.captions, model)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)
    with torch.no_grad():
        acc1 = test_epoch_blink(model,testloader,text_embeddings)
        print("=====================================")
        print(f"BLINK : {acc1}")
        print("=====================================")
    totalacc += acc1*len(testset)
    count += len(testset)
    return

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
    test_blink(model)
    

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





