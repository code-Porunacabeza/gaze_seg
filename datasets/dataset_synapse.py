import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk
from pywt import dwt2

def random_rot_flip(data):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    out=[]
    for d in data:
        d=np.rot90(d,k)
        d = np.flip(d, axis=axis).copy()
        out.append(d)
    return out


def random_rotate(data):
    angle = np.random.randint(-20, 20)
    out=[]
    for d in data:
        d = ndimage.rotate(d, angle, order=0, reshape=False)
        out.append(d)
    return out


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        heatmap, gaze = sample['heatmap'], sample['gaze']
        ll, (cH, cV, cD) = dwt2(heatmap, 'haar')
        ac = cv2.normalize(ll, None, alpha=0, beta=1,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ac[np.where(ac < 0)] = 0
        if random.random() > 0.5:
            data=[image,label,gaze,ac,cH,cV,cD]
            data=random_rot_flip(data)
            image=data[0]
            label=data[1]
            gaze=data[2]
            ac=data[3]
            cH=data[4]
            cV=data[5]
            cD=data[6]
        elif random.random() > 0.5:
            data = [image, label, gaze, ac, cH, cV, cD]
            data = random_rotate(data)
            image = data[0]
            label = data[1]
            gaze = data[2]
            ac = data[3]
            cH = data[4]
            cV = data[5]
            cD = data[6]
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            gaze = zoom(gaze, (self.output_size[0] / x, self.output_size[1] / y,1), order=3)
            ac = zoom(ac, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            cH = zoom(cH, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            cV = zoom(cV, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            cD = zoom(cD, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        
        cH=cH[np.newaxis,:,:]
        cV=cV[np.newaxis,:,:]
        cD=cD[np.newaxis,:,:]
        dc=np.concatenate((cH,cV,cD),axis=0)
        gaze = gaze.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        ac = torch.from_numpy(ac.astype(np.float32)).unsqueeze(0)
        gaze = torch.from_numpy(gaze.astype(np.float32))
        dc=torch.from_numpy(dc.astype(np.float32))
        sample = {'image': image, 'label': label.long() ,'ac':ac , 'gaze':gaze, 'dc':dc}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            heatmap_path=os.path.join(self.data_dir,'heatmap',slice_name+'_heatmap.png')
            gaze_path = os.path.join(self.data_dir, 'gaze', slice_name + '_gaze.npy')

            if not os.path.exists(heatmap_path):
                heatmap_path=os.path.join(self.data_dir,'heatmap','background_heatmap.png')
                gaze_path = os.path.join(self.data_dir, 'gaze', 'background_gaze.npy')
            image, label = data['image'], data['label']
            heatmap=cv2.imread(heatmap_path,0)
            gaze=np.load(gaze_path)


        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            gaze_path=self.data_dir + "/{}_gaze.nii.gz".format(vol_name)
            heat_path = self.data_dir + "/{}_heatmap.npy".format(vol_name)
            data = h5py.File(filepath)
            gaze_itk=sitk.ReadImage(gaze_path)
            gaze=sitk.GetArrayFromImage(gaze_itk)
            heatmap=np.load(heat_path)
            image, label = data['image'][:], data['label'][:]


        sample = {'image': image, 'label': label ,'heatmap':heatmap ,'gaze':gaze}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
