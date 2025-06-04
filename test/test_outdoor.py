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
from PIL import Image
import argparse
import json
import torch
device = torch.device("cuda:2")  

class RGBD:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.types = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['id']] = data['captions']
                    self.imgs[data['id']] = data['image']
                    self.answers[data['id']] = data['answer']
                    self.types[data['id']] = data['type']

class RGBD_Outdoor_Benchmark(Dataset):
    def __init__(self, root_dir,tasks):
        self.root_dir = root_dir
        self.dataset = RGBD(os.path.join(root_dir, tasks))
        self.image_ids = list(self.dataset.imgs.keys())
        self.captions = [x for x in self.dataset.anns.values()]
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.transform =transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'pic_all', self.dataset.imgs[img_ids])
        image = Image.open(image_path).convert('RGB')
        # depth = Image.open(depth_path).convert('L')

        answer = self.dataset.answers[img_ids]

        if self.transform:
            image = self.transform(image)
            depth = depth_model(image.unsqueeze(0))
            depth = depth.squeeze(0)
        return image, depth, answer

class RGBD_INDOOR:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.scene_id = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['id']] = data['captions']
                    self.imgs[data['id']] = data['image']
                    self.answers[data['id']] = data['answer']
                    self.scene_id[data['id']] = data['scene_id']

class RGBD_Indoor_Benchmark(Dataset):
    def __init__(self):
        self.root_dir = '../dataset/spatial-benchmark/indoors/pic_all'
        self.dataset = RGBD_INDOOR('../dataset/spatial-benchmark/indoors/all_data.json')
        self.image_ids = list(self.dataset.imgs.keys())
        self.captions = [x for x in self.dataset.anns.values()]
        self.transform =transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'scannet_2d_HR3', self.dataset.scene_id[img_ids],'color',self.dataset.imgs[img_ids])
        image = Image.open(image_path).convert('RGB')
        
        answer = self.dataset.answers[img_ids]

        if self.transform:
            image = self.transform(image)
            depth = depth_model(image.unsqueeze(0))
            depth = depth.squeeze(0)
        return image, depth, answer


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.eval()


def zeroshot_classifier_ours(captions, model):
    with torch.no_grad():
        zeroshot_weights = []
        for caption in tqdm(captions):
            texts = clip.tokenize(caption).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    return zeroshot_weights

def test_epoch_ours(dataloader,model,text_embeddings):
    corr_pred = 0
    total_num = 0
    for i, (images, depth, answer) in enumerate(tqdm(dataloader)):
        images = images.cuda()
        answer = answer.cuda()
        depth = depth.cuda()
        image_features = model.visual(images, depth,pos_embed="")
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = image_features.unsqueeze(1).repeat([1, 4, 1]) * text_embeddings
        pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
        corr_pred += torch.eq(pred, answer).sum()
        print(torch.eq(pred, answer))
        total_num += images.size(0)
    return corr_pred / total_num

def test_dataset(testset,model, dataset_name):
        print(f"Testing {dataset_name} dataset...")
        text_embeddings = zeroshot_classifier_ours(testset.captions, model)
        sampler = SequentialSampler(testset)
        testloader = DataLoader(testset, batch_size=256, sampler=sampler, pin_memory=True)
        with torch.no_grad():
            accuracy = test_epoch_ours(testloader, model, text_embeddings)
            print(f"{dataset_name} accuracy: {accuracy:.4f}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = sclip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)
    resume_pth = '../checkpoint/iter_5000.pth'
    ckpt = torch.load(resume_pth, map_location="cpu")
    new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    info=model.load_state_dict(new_ckpt, strict=True)
    print(info)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  

    model.to(device)
    testset_outdoor = RGBD_Outdoor_Benchmark('../dataset/spatial-benchmark/outdoors', 'all_data.json')
    test_dataset(testset_outdoor, model,"RGBD_Outdoor_Benchmark")

    testset_indoor = RGBD_Indoor_Benchmark()
    test_dataset(testset_indoor, model,"RGBD_Indoor_Benchmark")
    

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

