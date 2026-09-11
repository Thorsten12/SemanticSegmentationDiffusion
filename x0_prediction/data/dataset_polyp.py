import os
import random
import numpy as np
from collections import OrderedDict
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils import data
from torchvision.transforms import v2
from torchvision import transforms as T
from torchvision import tv_tensors

__all__ = ['PolypDataset']




def build_polyp_transforms(image_size, train: bool, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    geom = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # v2.RandomRotation([0, 90, 180, 270]),            # random angle in [-90,90]
        v2.RandomApply([v2.RandomRotation(90)], p=0.5),  # 90 degree rotation with 50% chance
        v2.Resize((image_size, image_size)),
    ]
    if not train:
        # for val/test: just resize
        geom = [v2.Resize((image_size, image_size))]

    geom_transforms = v2.Compose(geom)
    
    # img_post = T.Compose([
    #     T.ToTensor(),
    #     T.Lambda(lambda x: (x-x.min())/(x.max()-x.min() + 1e-8)),  # convert to float32 after ToTensor (which gives uint8)
    # ])
    img_post = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),      # [0,1] float32
        v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # normalize to [0,1]
    ])

    # img_post = v2.Compose([
    #     v2.ToDtype(torch.float32, scale=True),      # [0,1] float32
    #     v2.Normalize(mean=mean, std=std),
    # ])

    return geom_transforms, img_post


class PolypDatasetFast(data.Dataset):
    def __init__(self, image_root, gt_root, image_size=352, train=True, verbose=False):
        self.image_size = image_size
        self.is_train = bool(train)

        self.images = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.gts = sorted([
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.lower().endswith(('.tif', '.png'))
        ])

        if verbose: print("loading images and masks into memory...")
        self.filter_and_load_files()
        self.size = len(self.images)
        if verbose: print(f"  loaded {self.size} valid image-mask pairs.")

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self._set_train(self.is_train)
        self.inv_norm = self.make_inverse_img_norm()

    def __len__(self):
        return self.size
    
    def _set_train(self, train: bool):
        self.is_train = bool(train)
        self.geom_transforms, self.img_post = build_polyp_transforms(
            self.image_size, train=self.is_train, mean=self.mean, std=self.std
        )
    def eval(self):
        self._set_train(False)
    def train(self):
        self._set_train(True)
    
    def make_inverse_img_norm(self):
        inv_mean = [-m/s for m, s in zip(self.mean, self.std)]
        inv_std = [1.0/s for s in self.std]
        inv_normalize = v2.Normalize(mean=inv_mean, std=inv_std)
        return inv_normalize
    
    def make_pil_img(self, img_tensor):
        img_tensor = self.inv_norm(img_tensor)
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        return pil_img

    # ---------- helpers ---------- #
    def post_load_pil2tv2(self, pil_img, pil_gt):
        img = torch.from_numpy(np.array(pil_img))          # (H, W, 3), uint8
        img = img.permute(2, 0, 1)                         # (3, H, W)

        mask = torch.from_numpy(np.array(pil_gt))          # (H, W), uint8

        # wrap into TVTensors so v2 knows "this is an Image" / "this is a Mask"
        img  = tv_tensors.Image(img)                       # dtype uint8
        mask = tv_tensors.Mask(mask)                       # dtype uint8

        return img, mask
    
    def filter_and_load_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        self.im_paths, self.gt_paths = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                self.im_paths.append(img_path)
                self.gt_paths.append(gt_path)
                
                im, gt = self.post_load_pil2tv2(img.convert("RGB"), gt.convert("L"))
                images.append(im)
                gts.append(gt)
            else:
                print(f"Wrong pair (size mismatch): {img_path}, {gt_path}")
        self.images = images
        self.gts = gts

    def __getitem__(self, idx):
        # ---- load as numpy / tensors ---- #
        img = self.images[idx]
        msk  = self.gts[idx]
        im_path = self.im_paths[idx]
        gt_path = self.gt_paths[idx]

        # transforms (and AUGMENTATION for train set)
        img, msk = self.geom_transforms(img, msk)

        # image-only normalization
        img = self.img_post(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # ensure [0,1] range after transforms
        
        # convert msk to [0,1] float and add channel dim
        msk = (msk > 0).long()   # works for uint8 masks

        data = {
            'image': img,
            'label': msk,
            'id': idx,
            'im_path': im_path,
            'gt_path': gt_path,
        }
        return data


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, image_size, augmentations):
        self.image_size = image_size
        self.augmentations = augmentations
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = T.Compose([
                T.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = T.Compose([
                T.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.image_size or w < self.image_size:
            h = max(h, self.image_size)
            w = max(w, self.image_size)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_polyp(args, 
              test_folders = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir", "test"],
              only_test=False, only_train=False, tr_val_ratio = 0.2,
              target_test: str|None = "CVC-ClinicDB",
              logger=None, verbose=False):
    
    tr_dataset, vl_dataset = [], []
    if only_train:
        tr_dataset = PolypDatasetFast(image_root=f"{args.data_dir}/TrainDataset/images/", 
                                    gt_root=f"{args.data_dir}/TrainDataset/masks/", 
                                    image_size=args.img_size, train=True, verbose=True)
        vl_dataset = []
        te_dataset = []

    else:
        if not only_test:
            full_tr_dataset = PolypDatasetFast(image_root=f"{args.data_dir}/TrainDataset/images/", 
                                        gt_root=f"{args.data_dir}/TrainDataset/masks/", 
                                        image_size=args.img_size, train=True, verbose=True)
            # Split to training and validation sets
            
            val_size = int(len(full_tr_dataset) * tr_val_ratio)
            tr_size = len(full_tr_dataset) - val_size
            tr_dataset, vl_dataset = random_split(full_tr_dataset, [tr_size, val_size])
            vl_dataset.dataset.eval()  # set validation dataset to eval mode

        test_datasets = OrderedDict()
        for folder in test_folders:
            te_dataset = PolypDatasetFast(image_root=f"{args.data_dir}/TestDataset/{folder}/images/", 
                                gt_root=f"{args.data_dir}/TestDataset/{folder}/masks/", 
                                image_size=args.img_size, train=False, verbose=True)
            test_datasets[folder] = te_dataset
        te_dataset = []
        if target_test is not None:
            te_dataset = test_datasets[target_test]  # example for one test dataset

    if logger is not None:
        logger.info(f"PolypDataset: loaded samples from {args.data_dir}")
        logger.info(f"PolypDataset: training set size: {len(tr_dataset)}")
        logger.info(f"PolypDataset: validation set size: {len(vl_dataset)}")
        logger.info(f"PolypDataset: test set size ({target_test}): {len(te_dataset)}")

    if verbose:
        print("Polyp:")
        print(f"├──> Length of trainig_dataset:\t   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset ({target_test}):\t   {len(te_dataset)}")

    return {
        "tr_dataset": tr_dataset if not only_test else None,
        "vl_dataset": vl_dataset if not only_test else None,
        "te_dataset": te_dataset if not only_train else None,
        "te_datasets": test_datasets if target_test is None else None,
    }



# def get_loader(image_root, gt_root, batch_size, image_size, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

#     dataset = PolypDataset(image_root, gt_root, image_size, augmentation)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batch_size,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader


# class test_dataset:
#     def __init__(self, image_root, gt_root, testsize):
#         self.testsize = testsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.transform = transforms.Compose([
#             transforms.Resize((self.testsize, self.testsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.ToTensor()
#         self.size = len(self.images)
#         self.index = 0

#     def load_data(self):
#         image = self.rgb_loader(self.images[self.index])
#         image = self.transform(image).unsqueeze(0)
#         gt = self.binary_loader(self.gts[self.index])
#         name = self.images[self.index].split('/')[-1]
#         if name.endswith('.jpg'):
#             name = name.split('.jpg')[0] + '.png'
#         self.index += 1
#         return image, gt, name

#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')

#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')
