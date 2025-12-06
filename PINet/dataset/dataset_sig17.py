#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *

class SIG17_Training_Dataset(Dataset):

    def __init__(self, root_dir, sub_set, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.sub_set = sub_set

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        expoTimes = []
        with open(self.image_list[index][0]) as lines:
            for line in lines:
                if line[0]=='-':
                    expoTimes.append(int(line[1])*(-1))
                else:
                    expoTimes.append(int(line[0]))

        ldr_images = read_images(self.image_list[index][1])

        label = read_label(self.image_list[index][2], 'HDRImg.hdr')

        pre_img0 = gamma_correction(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = gamma_correction(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = gamma_correction(ldr_images[2], expoTimes[2], 2.2)

        pre_img3 = gamma_correction(ldr_images[0], expoTimes[0], 0.46)
        pre_img4 = gamma_correction(ldr_images[1], expoTimes[1], 0.46)
        pre_img5 = gamma_correction(ldr_images[2], expoTimes[2], 0.46)

        pre_img0 = np.concatenate((ldr_images[0], pre_img0, pre_img3), 2)
        pre_img1 = np.concatenate((ldr_images[1], pre_img1, pre_img4), 2)
        pre_img2 = np.concatenate((ldr_images[2], pre_img2, pre_img5), 2)

        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            'label': label
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)

class SIG17_Validation_Dataset(Dataset):

    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        self.scenes_dir = osp.join(root_dir, 'Test')
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        expoTimes = []
        with open(self.image_list[index][0]) as lines:
            for line in lines:
                if line[0] == '-':
                    expoTimes.append(int(line[1]) * (-1))
                else:
                    expoTimes.append(int(line[0]))

        ldr_images = read_images(self.image_list[index][1])

        label = read_label(self.image_list[index][2], 'HDRImg.hdr')

        pre_img0 = gamma_correction(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = gamma_correction(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = gamma_correction(ldr_images[2], expoTimes[2], 2.2)

        pre_img3 = gamma_correction(ldr_images[0], expoTimes[0], 0.46)
        pre_img4 = gamma_correction(ldr_images[1], expoTimes[1], 0.46)
        pre_img5 = gamma_correction(ldr_images[2], expoTimes[2], 0.46)

        pre_img0 = np.concatenate((ldr_images[0], pre_img0, pre_img3), 2)
        pre_img1 = np.concatenate((ldr_images[1], pre_img1, pre_img4), 2)
        pre_img2 = np.concatenate((ldr_images[2], pre_img2, pre_img5), 2)


        if self.crop:
            x = 0
            y = 0
            img0 = pre_img0[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            label = label[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
        else:
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0,
            'input1': img1,
            'input2': img2,
            'label': label
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)
