import os 
import json
from albumentations.augmentations.functional import gauss_noise
from albumentations.augmentations.geometric.transforms import ElasticTransform
from albumentations.augmentations.transforms import CLAHE, HorizontalFlip, VerticalFlip
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

with open('/opt/ml/segmentation/input/data/val.json', 'r') as f:
    cates = json.load(f)['categories']

category_names = ['Background']
category_names.extend([x['name'] for x in cates])
print(category_names)
# exit()

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class WasteDataset(Dataset):
    def __init__(self, dataroot, mode='train', transform=None):
        super().__init__()
        self.dataroot = dataroot
        self.mode = mode
        self.transform = transform
        self.coco = COCO(os.path.join(dataroot, f'{mode}.json'))

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, index):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        imgs = np.array(Image.open(os.path.join(self.dataroot, image_infos['file_name'])))
        # print(imgs[0])
        # # exit()
        # print('#'*30)
        # print('\n'*5)

        if self.mode in ('train', 'val'):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            annots = self.coco.loadAnns(ann_ids)
            annots = sorted(annots, key=lambda idx : len(idx['segmentation'][0]), reverse=False)

            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            masks = np.zeros((image_infos['height'], image_infos['width']))
            for i in range(len(annots)):
                className = get_classname(annots[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(annots[i])==1] = pixel_value
            masks = masks.astype(np.int8)
            # print(masks)

            if self.transform:
                transformed = self.transform(image=imgs, mask=masks)
                imgs = transformed['image']
                masks = transformed['mask']
            # return {'image': imgs, 'mask': masks, 'info': image_infos}
            return imgs, masks, image_infos['file_name']

        if self.mode == 'test':
            if self.transform:
                transformed = self.transform(image=imgs)
                imgs = transformed['image']
            # return {'image': imgs, 'info': image_infos}
            return imgs, image_infos['file_name']


A_train = A.Compose([
    A.Normalize(mean=[0.46098824, 0.44022745, 0.41892157], std=[0.23398431, 0.23115294, 0.24377255]),
    A.Flip(p=0.5),
    A.Rotate(p=0.3),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.GaussNoise(p=0.3),
    # A.ElasticTransform(p=0.3),
    # A.CLAHE(p=0.4),
    # A.ChannelShuffle(p=0.3),
    # A.Blur(p=0.3),
    # A.RandomShadow(p=0.3),
    ToTensorV2(),
])
A_val = A.Compose([
    A.Normalize(mean=[0.46098824, 0.44022745, 0.41892157], std=[0.23398431, 0.23115294, 0.24377255]),
    ToTensorV2(),
])
A_test = A.Compose([
    A.Normalize(mean=[0.46098824, 0.44022745, 0.41892157], std=[0.23398431, 0.23115294, 0.24377255]),
    ToTensorV2(),
])

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloader(dataroot, mode, opt):
    if mode == 'train':
        train_dataset = WasteDataset(dataroot=dataroot, mode='train', transform=A_train)
        val_dataset = WasteDataset(dataroot=dataroot, mode='val', transform=A_val)
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=opt.workers,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=opt.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=opt.workers,
            collate_fn=collate_fn
        )
        return train_loader, val_loader

    elif mode == 'test':
        test_dataset = WasteDataset(dataroot=os.path.join(dataroot, 'test.json'), mode='test', transform=A_test)

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opt.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=opt.workers,
            collate_fn=collate_fn
        )
        return test_loader

    else:
        print(f'no mode named {opt.mode}')
        exit()