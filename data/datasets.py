import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
import shutil
# from data.datasets import get_data_loaders
import glob
import random
import itertools
import scipy.ndimage as ndimage

class SegDataset(Dataset):
    @staticmethod
    def norm_img(img):
        """标准化到[0,1]范围"""
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    def __init__(self, A_path, transforms=None,img_size=(192, 192, 80)):
        # self.dir_B = os.path.join(A_path, 'CT1.5/')
        self.dir_B = os.path.join(A_path, 'affine223CT/')
        # self.dir_B_seg = os.path.join(A_path, 'label1.5liver/')
        self.dir_B_seg = os.path.join(A_path, 'affine223liver/')
        self.B_paths = sorted(glob.glob(self.dir_B+'/*.nii.gz'))

        self.B_size = len(self.B_paths)
        self.transforms = transforms
        self.img_size = img_size

    def pad_crop(self, img):
        # img = self.norm_img(img)
        orig_dims = np.array(img.shape)
        target_dims = np.array(self.img_size)
        pad_width = np.maximum(target_dims - orig_dims, 0)
        pad_before = pad_width // 2
        pad_after = pad_width - pad_before
        # newdata = np.pad(img, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)),
        #                  mode='constant',
        #                  constant_values=0)
        padded_img = np.pad(img,((pad_before[0], pad_after[0]),
                                (pad_before[1], pad_after[1]),
                                (pad_before[2], pad_after[2])),
                                mode='constant',
                                constant_values=0)
        start_idx = (np.array(padded_img.shape) - target_dims) // 2
        cropped_img = padded_img[
                      start_idx[0]:start_idx[0] + target_dims[0],
                      start_idx[1]:start_idx[1] + target_dims[1],
                      ]
        cropped_img = cropped_img[:,:, 6:70]  # z轴为64
        return cropped_img

    def __getitem__(self, index):
        img1_path = self.B_paths[index]
        img1_name = os.path.basename(img1_path)
        img1_seg_path = os.path.join(self.dir_B_seg, img1_name)  # .replace('.nii','.nii.gz'))
    

        vol1 = nib.load(img1_path)
        A_seg = nib.load(img1_seg_path)

        img1 = vol1.get_fdata().astype(np.float32)
        img1_seg= A_seg.get_fdata().astype(np.float32)

     
        # 预处理
        img1 = self.pad_crop(img1)
        # img2 = self.pad_crop(img2)
        img1_seg = self.pad_crop(img1_seg)
        # img2_seg = self.pad_crop(img2_seg)

        img1 = img1[np.newaxis, ...]  # (1, H, W, D)
        # img2 = img2[np.newaxis, ...]
        img1_seg = img1_seg[np.newaxis, ...]
        # img2_seg = img2_seg[np.newaxis, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        if self.transforms:
            img1 = self.transforms(img1)
        img1 = torch.from_numpy(img1.copy()).permute(0, 3, 1, 2)  # (1, D, H, W)
        # img2 = torch.from_numpy(img2.copy()).permute(0, 3, 1, 2)
        img1_seg = torch.from_numpy(img1_seg.copy()).permute(0, 3, 1, 2)  # (1, D, H, W)
        # img2_seg = torch.from_numpy(img2_seg.copy()).permute(0, 3, 1, 2)
        return img1,img1_seg #,img1_name

    def __len__(self):
        return self.B_size


class multiSegDataset(Dataset):
    @staticmethod
    def norm_img(img):
        """标准化到[0,1]范围"""
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    def __init__(self, A_path, transforms=None, img_size=(192, 192, 80),augment=False):
        # self.dir_B = os.path.join(A_path, 'CT1.5/')
        self.dir_B = os.path.join(A_path, 'affine223CT/')
        # self.dir_B_seg = os.path.join(A_path, 'label1.5/')
        self.dir_B_seg = os.path.join(A_path, 'affinelabel/')
        self.B_paths = sorted(glob.glob(self.dir_B + '/*.nii.gz'))
        self.keep_labels = {1: 1, 2: 2, 3: 3, 6: 4}

        self.B_size = len(self.B_paths)
        # self.transforms = transforms
        self.img_size = img_size

        self.augment = augment
    def pad_crop(self, img):
        # img = self.norm_img(img)
        orig_dims = np.array(img.shape)
        target_dims = np.array(self.img_size)
        pad_width = np.maximum(target_dims - orig_dims, 0)
        pad_before = pad_width // 2
        pad_after = pad_width - pad_before
        # newdata = np.pad(img, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)),
        #                  mode='constant',
        #                  constant_values=0)
        padded_img = np.pad(img, ((pad_before[0], pad_after[0]),
                                  (pad_before[1], pad_after[1]),
                                  (pad_before[2], pad_after[2])),
                            mode='constant',
                            constant_values=0)
        start_idx = (np.array(padded_img.shape) - target_dims) // 2
        cropped_img = padded_img[
            start_idx[0]:start_idx[0] + target_dims[0],
            start_idx[1]:start_idx[1] + target_dims[1],
        ]
        cropped_img = cropped_img[:, :, 6:70]  # z轴为64
        return cropped_img

    def remap_labels(self, lbl):
        new_lbl = np.zeros_like(lbl, dtype=np.int16)
        for orig, new in self.keep_labels.items():
            new_lbl[lbl == orig] = new
        return new_lbl

    #----------------数据增强-----------------------------
    def random_flip(self, img, lbl):
        # 随机在 x/y 方向翻转；z 一般不翻或少翻
        # 你可以按自己的经验调整哪些轴可以翻转
        if random.random() < 0.5:
            img = np.flip(img, axis=0)
            lbl = np.flip(lbl, axis=0)
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            lbl = np.flip(lbl, axis=1)
        return img, lbl

    # def random_scale(self, img, lbl, scale_range=(0.9, 1.1)):
    #     if random.random() < 0.5:
    #         scale_x = random.uniform(scale_range[0], scale_range[1])
    #         scale_y = random.uniform(scale_range[0], scale_range[1])
    #         scale_z = random.uniform(scale_range[0], scale_range[1])
    #
    #         zoom_factors = (scale_x, scale_y, scale_z)
    #         # 图像
    #         img = ndimage.zoom(img, zoom_factors, order=3)
    #         # 标签
    #         lbl = ndimage.zoom(lbl, zoom_factors, order=0)
    #         # 再次 pad_crop 回到固定大小
    #         img = self.pad_crop(img)
    #         lbl = self.pad_crop(lbl)
    #     return img, lbl
    def random_intensity(self, img):
        """只对图像做强度增强"""
        # 随机亮度偏移
        if random.random() < 0.5:
            # 将随机数转换为 float32
            shift = np.float32(random.uniform(-0.1, 0.1))
            img = img + shift
        # 随机对比度
        if random.random() < 0.5:
            # 将随机数转换为 float32
            factor = np.float32(random.uniform(0.9, 1.1))
            mean_val = img.mean()
            img = (img - mean_val) * factor + mean_val
        # Gamma
        if random.random() < 0.5:
            # 将随机数转换为 float32
            gamma = np.float32(random.uniform(0.7, 1.3))
            # 在做幂运算前，确保img的值为正
            img = np.clip(img, 1e-7, 1.0)  # clip(0,1)可能导致log(0)等问题，用一个很小的值代替0
            img = img ** gamma
        # 高斯噪声
        if random.random() < 0.3:
            sigma = random.uniform(0.0, 0.05)
            # 明确指定噪声的数据类型为 float32
            noise = np.random.normal(0, sigma, size=img.shape).astype(np.float32)
            img = img + noise

        # 保证仍在 [0,1] 左右，防止过度失真
        img = np.clip(img, 0, 1)

        # 最后再保险一步，确保返回的是 float32
        return img.astype(np.float32)

    def augment_sample(self, img, lbl):
        """组合增强流程：几何 + 强度"""
        # 几何增强（必须 img/lbl 同时变）
        img, lbl = self.random_flip(img, lbl)
        # img, lbl = self.random_scale(img, lbl, scale_range=(0.9, 1.1))
        # 强度增强（只对 img）
        img = self.random_intensity(img)
        return img, lbl

    def __getitem__(self, index):
        img1_path = self.B_paths[index]
        img1_name = os.path.basename(img1_path)
        img1_seg_path = os.path.join(self.dir_B_seg, img1_name)  # .replace('.nii','.nii.gz'))

        vol1 = nib.load(img1_path)
        A_seg = nib.load(img1_seg_path)

        img1 = vol1.get_fdata().astype(np.float32)
        img1_seg = A_seg.get_fdata().astype(np.int64)

        # 预处理
        img1 = self.pad_crop(img1)
        # img2 = self.pad_crop(img2)
        img1_seg = self.pad_crop(img1_seg)
        # img2_seg = self.pad_crop(img2_seg)
        #多器官映射
        img1_seg = self.remap_labels(img1_seg)


        if self.augment:
            # print(img1.shape)
            img1, img1_seg = self.augment_sample(img1, img1_seg)

        img1 = img1[np.newaxis, ...]  # (1, H, W, D)
        # img2 = img2[np.newaxis, ...]
        img1_seg = img1_seg[np.newaxis, ...]
        # img2_seg = img2_seg[np.newaxis, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)

        # img1 = torch.from_numpy(img1.copy()).permute(0, 3, 1, 2)  # (1, D, H, W)
        img1 = torch.from_numpy(np.ascontiguousarray(img1)).permute(0, 3, 1, 2)  # (1, D, H, W)
        # img2 = torch.from_numpy(img2.copy()).permute(0, 3, 1, 2)
        # img1_seg = torch.from_numpy(img1_seg.copy()).permute(0, 3, 1, 2)  # (1, D, H, W)
        img1_seg = torch.from_numpy(np.ascontiguousarray(img1_seg)).permute(0, 3, 1, 2)  # (1, D, H, W)
        # img2_seg = torch.from_numpy(img2_seg.copy()).permute(0, 3, 1, 2)
        return img1, img1_seg  # ,img1_name

    def __len__(self):
        return self.B_size



class RegDataset(Dataset):
    def __init__(self, A_path, Atlas,transforms=None, img_size=(192, 192, 100)):
        self.dir_B = os.path.join(A_path, 'affine223CT/')
        # self.dir_B = os.path.join(A_path, 'CT1.5/')
        # self.dir_B_seg = os.path.join(A_path, 'affine223liver/')
        self.B_paths = glob.glob(self.dir_B + '/*.nii.gz')
        # self.file_pairs = list(itertools.permutations(self.B_paths, 2))
        self.atlas = Atlas


        self.B_size = len(self.B_paths)
        self.transforms = transforms
        self.img_size = img_size

    def pad_crop(self, img):
        orig_dims = np.array(img.shape)
        target_dims = np.array(self.img_size)
        pad_width = np.maximum(target_dims - orig_dims, 0)
        pad_before = pad_width // 2
        pad_after = pad_width - pad_before
        padded_img = np.pad(img, ((pad_before[0], pad_after[0]),
                                  (pad_before[1], pad_after[1]),
                                  (pad_before[2], pad_after[2])),
                            mode='constant',
                            constant_values=0)
        start_idx = (np.array(padded_img.shape) - target_dims) // 2
        cropped_img = padded_img[
            start_idx[0]:start_idx[0] + target_dims[0],
            start_idx[1]:start_idx[1] + target_dims[1],
        ]
        cropped_img = cropped_img[:, :, 5:69]  # z轴为64
        return cropped_img


    def __getitem__(self, index):
        img2_path = self.atlas
        img1_path = self.B_paths[index]
        # num = random.randint(0, 150)
        # img2_path = self.B_paths[(index+num)%self.B_size]
        # img1_path, img2_path = self.file_pairs[index]

        # img1_name = os.path.basename(self.atlas)
        # img2_name = os.path.basename(img2_path)

        # img1_seg_path = os.path.join(self.dir_B_seg, img1_name)  # .replace('.nii','.nii.gz'))
        # img2_seg_path = os.path.join(self.dir_B_seg, img2_name)  # .replace('.nii','.nii.gz'))

        vol1 = nib.load(img1_path)
        vol2 = nib.load(img2_path)
        # A_seg = nib.load(img1_seg_path)
        # B_seg = nib.load(img2_seg_path)

        img1 = vol1.get_fdata().astype(np.float32)
        img2 = vol2.get_fdata().astype(np.float32)
        # img1_seg = A_seg.get_fdata().astype(np.float32)
        # img2_seg = B_seg.get_fdata().astype(np.float32)

        # 预处理
        img1 = self.pad_crop(img1)
        img2 = self.pad_crop(img2)
        # img1_seg = self.pad_crop(img1_seg)
        # img2_seg = self.pad_crop(img2_seg)

        img1 = img1[np.newaxis, ...]  # (1, H, W, D)
        img2 = img2[np.newaxis, ...]
        # img1_seg = img1_seg[np.newaxis, ...]
        # img2_seg = img2_seg[np.newaxis, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        # if self.transforms:
        #     img1 = self.transforms(img1)
        img1 = torch.from_numpy(img1.copy()).permute(0, 3, 1, 2)  # (1, D, H, W)
        img2 = torch.from_numpy(img2.copy()).permute(0, 3, 1, 2)
        # img1_seg = torch.from_numpy(img1_seg.copy()).permute(0, 3, 1, 2)  # (1, D, H, W)
        # img2_seg = torch.from_numpy(img2_seg.copy()).permute(0, 3, 1, 2)
        return img1, img2 #, img1_seg, img2_seg  # ,img1_name

    def __len__(self):
        return self.B_size

class RegValDataset(Dataset):
    def __init__(self, A_path, Atlas,transforms=None, img_size=(192, 192, 100)):
        self.dir_B = os.path.join(A_path, 'affine223CT/')
        # self.dir_B = os.path.join(A_path, 'CT1.5/')
        self.dir_B_seg = os.path.join(A_path, 'affinelabel/')
        # self.dir_B_seg = os.path.join(A_path, 'label1.5_liver/')
        self.B_paths = glob.glob(self.dir_B + '/*.nii.gz')
        # self.file_pairs = list(itertools.permutations(self.B_paths, 2))
        self.atlas = Atlas
        self.keep_labels = {1: 1, 2: 2, 3: 3, 6: 4}

        self.B_size = len(self.B_paths)
        self.transforms = transforms
        self.img_size = img_size

    def pad_crop(self, img):
        orig_dims = np.array(img.shape)
        target_dims = np.array(self.img_size)
        pad_width = np.maximum(target_dims - orig_dims, 0)
        pad_before = pad_width // 2
        pad_after = pad_width - pad_before
        padded_img = np.pad(img, ((pad_before[0], pad_after[0]),
                                  (pad_before[1], pad_after[1]),
                                  (pad_before[2], pad_after[2])),
                            mode='constant',
                            constant_values=0)
        start_idx = (np.array(padded_img.shape) - target_dims) // 2
        cropped_img = padded_img[
            start_idx[0]:start_idx[0] + target_dims[0],
            start_idx[1]:start_idx[1] + target_dims[1],
        ]
        cropped_img = cropped_img[:, :, 5:69]  # z轴为64
        return cropped_img

    def remap_labels(self, lbl):
        new_lbl = np.zeros_like(lbl, dtype=np.int16)
        for orig, new in self.keep_labels.items():
            new_lbl[lbl == orig] = new
        return new_lbl

    def __getitem__(self, index):
        img2_path = self.atlas
        img1_path = self.B_paths[index]
        # num = random.randint(0, 150)
        # img2_path = self.B_paths[(index+num)%self.B_size]
        # img1_path, img2_path = self.file_pairs[index]

        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)

        img1_seg_path = os.path.join(self.dir_B_seg, img1_name)  # .replace('.nii','.nii.gz'))
        img2_seg_path = os.path.join(self.dir_B_seg, img2_name)  # .replace('.nii','.nii.gz'))

        vol1 = nib.load(img1_path)
        vol2 = nib.load(img2_path)
        A_seg = nib.load(img1_seg_path)
        B_seg = nib.load(img2_seg_path)

        img1 = vol1.get_fdata().astype(np.float32)
        img2 = vol2.get_fdata().astype(np.float32)
        img1_seg = A_seg.get_fdata().astype(np.int64)
        img2_seg = B_seg.get_fdata().astype(np.int64)

        # 预处理
        img1 = self.pad_crop(img1)
        img2 = self.pad_crop(img2)
        img1_seg = self.pad_crop(img1_seg)
        img2_seg = self.pad_crop(img2_seg)

        img1_seg = self.remap_labels(img1_seg)
        img2_seg = self.remap_labels(img2_seg)

        img1 = img1[np.newaxis, ...]  # (1, H, W, D)
        img2 = img2[np.newaxis, ...]
        img1_seg = img1_seg[np.newaxis, ...]
        img2_seg = img2_seg[np.newaxis, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        # if self.transforms:
        #     img1 = self.transforms(img1)
        img1 = torch.from_numpy(np.ascontiguousarray(img1)).permute(0, 3, 1, 2)  # (1, D, H, W)
        img2 = torch.from_numpy(np.ascontiguousarray(img2)).permute(0, 3, 1, 2)
        img1_seg = torch.from_numpy(np.ascontiguousarray(img1_seg)).permute(0, 3, 1, 2)  # (1, D, H, W)
        img2_seg = torch.from_numpy(np.ascontiguousarray(img2_seg)).permute(0, 3, 1, 2)
        return img1, img2, img1_seg, img2_seg ,img1_name

    def __len__(self):
        return self.B_size


class AMOSDataset(Dataset):
    @staticmethod
    def norm_img(img):
        """标准化到[0,1]范围"""
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    def __init__(self, A_path, transforms=None,img_size=(160, 160, 80)):
        self.dir_B = os.path.join(A_path, '223CT/')
        self.dir_B_seg = os.path.join(A_path, '223CTlabel/')
        self.B_paths = sorted(glob.glob(self.dir_B+'/*.nii.gz'))
        self.file_pairs = list(itertools.permutations(self.B_paths, 2))

        print(len(self.file_pairs))
        self.B_size = len(self.B_paths)
        self.transforms = transforms
        self.img_size = img_size

    def pad_crop(self, img):
        img = self.norm_img(img)
        orig_dims = np.array(img.shape)
        target_dims = np.array(self.img_size)
        pad_width = np.maximum(target_dims - orig_dims, 0)
        pad_before = pad_width // 2
        pad_after = pad_width - pad_before
        # newdata = np.pad(img, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)),
        #                  mode='constant',
        #                  constant_values=0)
        padded_img = np.pad(img,((pad_before[0], pad_after[0]),
                                (pad_before[1], pad_after[1]),
                                (pad_before[2], pad_after[2])),
                                mode='constant',
                                constant_values=0)
        start_idx = (np.array(padded_img.shape) - target_dims) // 2
        cropped_img = padded_img[
                      start_idx[0]:start_idx[0] + target_dims[0],
                      start_idx[1]:start_idx[1] + target_dims[1],
                      ]
        cropped_img = cropped_img[:,:, 6:70]  # z轴为64
        return cropped_img

    def __getitem__(self, index):
        # A_path = self.file_pairs[index][0]
        img1_path, img2_path = self.file_pairs[index]
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)
        img1_seg_path = os.path.join(self.dir_B_seg, img1_name)  # .replace('.nii','.nii.gz'))
        img2_seg_path = os.path.join(self.dir_B_seg, img2_name)

        vol1 = nib.load(img1_path)
        vol2 = nib.load(img2_path)
        A_seg = nib.load(img1_seg_path)
        B_seg = nib.load(img2_seg_path)
        img1 = vol1.get_fdata().astype(np.float32)
        img2 = vol2.get_fdata().astype(np.float32)
        img1_seg= A_seg.get_fdata().astype(np.float32)
        img2_seg =B_seg.get_fdata().astype(np.float32)
        # 预处理
        img1 = self.pad_crop(img1)
        img2 = self.pad_crop(img2)
        img1_seg = self.pad_crop(img1_seg)
        img2_seg = self.pad_crop(img2_seg)

        img1 = img1[np.newaxis, ...]  # (1, H, W, D)
        img2 = img2[np.newaxis, ...]
        img1_seg = img1_seg[np.newaxis, ...]
        img2_seg = img2_seg[np.newaxis, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        if self.transforms:
            img1, img2 = self.transforms([img1, img2])
        img1 = torch.from_numpy(img1.copy()).permute(0, 3, 1, 2)  # (1, D, H, W)
        img2 = torch.from_numpy(img2.copy()).permute(0, 3, 1, 2)
        img1_seg = torch.from_numpy(img1_seg.copy()).permute(0, 3, 1, 2)  # (1, D, H, W)
        img2_seg = torch.from_numpy(img2_seg.copy()).permute(0, 3, 1, 2)
        return (img1,img2,img1_seg,img2_seg,img1_name,img2_name)

    def __len__(self):
        return len(self.file_pairs)


class MedicalDataset(Dataset):
    """Dataset for preprocessed medical images and labels"""
    
    def __init__(self, image_paths, label_paths, zscore_params=None, 
                 expected_shape=(224, 224, 64), target_organ=6,
                 save_invalid_dir=None):
        """
        Args:
            image_paths: List of paths to preprocessed image files
            label_paths: List of paths to preprocessed label files
            zscore_params: Optional dict with 'mean' and 'std' for z-score normalization
            expected_shape: Expected shape of images (x, y, z)
            target_organ: Label value for the organ to segment (6=liver in AMOS)
            save_invalid_dir: Directory to save corrected invalid images/labels (if not None)
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.zscore_params = zscore_params
        self.expected_shape = expected_shape
        self.target_organ = target_organ
        self.save_invalid_dir = save_invalid_dir
        self.invalid_pairs = []  # Track invalid image-label pairs
        
        # Verify file lists match
        if len(image_paths) != len(label_paths):
            logger.warning(f"Number of images ({len(image_paths)}) and labels ({len(label_paths)}) don't match")
        
        # Verify all files exist and have correct shape
        self.valid_indices = []
        for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
            if not os.path.exists(img_path):
                logger.error(f"Image file not found: {img_path}")
                continue
            if not os.path.exists(lbl_path):
                logger.error(f"Label file not found: {lbl_path}")
                continue
                
            try:
                # Check image shape
                img = nib.load(img_path).get_fdata()
                img_shape = img.shape
                
                # Check label shape
                lbl = nib.load(lbl_path).get_fdata()
                lbl_shape = lbl.shape
                
                # Check if shapes match expected dimensions
                if img_shape != self.expected_shape or lbl_shape != self.expected_shape:
                    # Store invalid pair for potential correction
                    self.invalid_pairs.append((img_path, lbl_path, img_shape, lbl_shape))
                    logger.warning(f"Image {img_path} has shape {img_shape}, expected {self.expected_shape}")
                    continue
                
                self.valid_indices.append(i)
            except Exception as e:
                logger.error(f"Error loading {img_path} or {lbl_path}: {str(e)}")
        
        logger.info(f"Found {len(self.valid_indices)} valid image-label pairs out of {len(image_paths)}")
        
        # Calculate dataset statistics if not provided
        if zscore_params is None and self.valid_indices:
            logger.info("Calculating z-score normalization parameters...")
            self.zscore_params = self._calculate_zscore_stats()
        
        # Save corrected invalid images if requested
        if save_invalid_dir and self.invalid_pairs:
            self._save_corrected_invalid_pairs(save_invalid_dir)
    
    def _calculate_zscore_stats(self):
        """Calculate mean and std for z-score normalization across dataset"""
        means = []
        stds = []
        
        for idx in self.valid_indices:
            img_path = self.image_paths[idx]
            img = self._load_nifti(img_path)
            # Only consider non-zero (anatomical) regions
            mask = img > 0
            if np.sum(mask) > 0:
                means.append(np.mean(img[mask]))
                stds.append(np.std(img[mask]))
        
        if not means:  # Handle case where no valid data found
            logger.error("No valid data found for calculating z-score stats. Using default values.")
            return {'mean': 0.0, 'std': 1.0}
            
        global_mean = np.mean(means)
        global_std = np.mean(stds)
        
        logger.info(f"Calculated z-score params - Mean: {global_mean:.4f}, Std: {global_std:.4f}")
        return {'mean': global_mean, 'std': global_std}
    
    def _load_nifti(self, path):
        """Load NIfTI file and return as numpy array"""
        img = nib.load(path)
        data = img.get_fdata()
        return data
    
    def _resize_and_save(self, input_path, output_path, is_label=False):
        """Resize image/label to expected shape and save to new location"""
        # Load the data
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Resize if needed
        if data.shape != self.expected_shape:
            logger.info(f"Resizing {input_path} from {data.shape} to {self.expected_shape}")
            factors = [
                self.expected_shape[0] / data.shape[0],
                self.expected_shape[1] / data.shape[1],
                self.expected_shape[2] / data.shape[2]
            ]
            
            if is_label:
                # For labels, use nearest neighbor interpolation
                data = zoom(data, factors, order=0)
            else:
                # For images, use linear interpolation
                data = zoom(data, factors, order=1)
        
        # Create NIfTI image with original affine
        new_img = nib.Nifti1Image(data, affine=img.affine)
        
        # Save to new location
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(new_img, output_path)
        logger.info(f"Saved corrected file to {output_path}")
        
        return data
    
    def _save_corrected_invalid_pairs(self, save_dir):
        """Save corrected invalid image-label pairs to the specified directory"""
        logger.info(f"Saving {len(self.invalid_pairs)} invalid image-label pairs to {save_dir}")
        
        # Create directory structure
        test_images_dir = os.path.join(save_dir, "images")
        test_labels_dir = os.path.join(save_dir, "labels")
        os.makedirs(test_images_dir, exist_ok=True)
        os.makedirs(test_labels_dir, exist_ok=True)
        
        # Process each invalid pair
        for img_path, lbl_path, img_shape, lbl_shape in self.invalid_pairs:
            # Get filename
            filename = os.path.basename(img_path)
            
            # Define output paths
            output_img_path = os.path.join(test_images_dir, filename)
            output_lbl_path = os.path.join(test_labels_dir, filename)
            
            # Resize and save image
            image_data = self._resize_and_save(img_path, output_img_path, is_label=False)
            
            # Resize and save label
            label_data = self._resize_and_save(lbl_path, output_lbl_path, is_label=True)
            
            # Convert label to binary for liver (label 6)
            label_data = (label_data == self.target_organ).astype(np.uint8)
            
            # Save binary label
            binary_label = nib.Nifti1Image(label_data, affine=nib.load(lbl_path).affine)
            binary_lbl_path = os.path.join(test_labels_dir, f"binary_{filename}")
            nib.save(binary_label, binary_lbl_path)
            logger.info(f"Saved binary label to {binary_lbl_path}")
        
        logger.info(f"Successfully saved corrected invalid pairs to {save_dir}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[idx]
        
        # Load image and label
        image = self._load_nifti(self.image_paths[actual_idx])
        label = self._load_nifti(self.label_paths[actual_idx])
        
        # Verify shapes match expected dimensions
        if image.shape != self.expected_shape:
            # This should not happen as we filtered during initialization
            image = self._resize_to_expected(image)
            
        if label.shape != self.expected_shape:
            # This should not happen as we filtered during initialization
            label = self._resize_to_expected(label, is_label=True)
        
        # CRITICAL FIX: Convert multi-class labels to binary for liver (label 6)
        # AMOS dataset: 6 = liver (confirmed by user)
        label = (label == self.target_organ).astype(np.uint8)
        
        # Apply z-score normalization
        image = (image - self.zscore_params['mean']) / (self.zscore_params['std'] + 1e-8)
        
        # Add channel dimension [C, D, H, W]
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        
        # Convert to torch tensors
        image = torch.FloatTensor(image)
        label = torch.LongTensor(label)
        
        return image, label
    
    def _resize_to_expected(self, array, is_label=False):
        """Resize array to expected shape using appropriate interpolation"""
        factors = [
            self.expected_shape[0] / array.shape[0],
            self.expected_shape[1] / array.shape[1],
            self.expected_shape[2] / array.shape[2]
        ]
        
        if is_label:
            # For labels, use nearest neighbor interpolation
            return zoom(array, factors, order=0)
        else:
            # For images, use linear interpolation
            return zoom(array, factors, order=1)

def get_data_loaders(train_img_dir, train_label_dir, val_img_dir, val_label_dir, 
                    batch_size=2, num_workers=4, zscore_params=None, 
                    expected_shape=(224, 224, 64), target_organ=6,
                    save_invalid_dir=None):
    """
    Create data loaders for training and validation
    
    Args:
        train_img_dir: Directory with training images
        train_label_dir: Directory with training labels
        val_img_dir: Directory with validation images
        val_label_dir: Directory with validation labels
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        zscore_params: Optional z-score parameters (mean, std)
        expected_shape: Expected shape of images (x, y, z)
        target_organ: Label value for the organ to segment (6=liver in AMOS)
        save_invalid_dir: Directory to save corrected invalid images/labels (if not None)
    
    Returns:
        train_loader, val_loader
    """
    # Get file lists
    train_image_files = sorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) 
                              if f.endswith('.nii.gz')])
    train_label_files = sorted([os.path.join(train_label_dir, f) for f in os.listdir(train_label_dir) 
                              if f.endswith('.nii.gz')])
    
    val_image_files = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) 
                            if f.endswith('.nii.gz')])
    val_label_files = sorted([os.path.join(val_label_dir, f) for f in os.listdir(val_label_dir) 
                            if f.endswith('.nii.gz')])
    
    # Create datasets
    train_dataset = MedicalDataset(
        train_image_files, 
        train_label_files, 
        zscore_params,
        expected_shape=expected_shape,
        target_organ=target_organ,
        save_invalid_dir=save_invalid_dir  # Pass the directory to save invalid pairs
    )
    
    # Use train dataset's zscore params for validation
    val_zscore_params = train_dataset.zscore_params if len(train_dataset) > 0 else None
    val_dataset = MedicalDataset(
        val_image_files, 
        val_label_files, 
        val_zscore_params,
        expected_shape=expected_shape,
        target_organ=target_organ,
        save_invalid_dir=None  # Don't save invalid pairs from validation again
    )
    
    # Handle case where no valid data found
    if len(train_dataset) == 0:
        logger.error("No valid training data found. Cannot create data loader.")
        raise ValueError("No valid training data found")
    
    if len(val_dataset) == 0:
        logger.warning("No valid validation data found.")
    
    # Create data loaders with proper error handling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} valid samples")
    logger.info(f"Validation dataset: {len(val_dataset)} valid samples")
    
    return train_loader, val_loader