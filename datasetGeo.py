import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import DataLoader, Dataset
import torch
import cv2 as cv
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import glob
import random
from osgeo import gdal
from skimage import io
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv.filter2D(image, -1, kernel=kernel)
    return dst


def read_geoimg(path):
    data = gdal.Open(path)
    lastChannel = data.RasterCount + 1
    arr = [data.GetRasterBand(idx).ReadAsArray() for idx in range(1, lastChannel)]
    arr = np.dstack(arr)
    return arr


class RandomFlip:
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            for k, v in sample.items():
                sample[k] = np.flip(v, d)

        return sample


class RandomRotate90:
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            for k, v in sample.items():
                sample[k] = np.rot90(v, factor)

        return sample


class Rescale(object):
    def __init__(self, output_size, prob=0.9):
        self.prob = prob
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        if random.random() < self.prob:
            image = sample["image"]
            raw_h, raw_w = image.shape[:2]
            for k, v in sample.items():
                sample[k] = cv.resize(v, (self.output_size, self.output_size))

            img = sample["image"]
            h, w = img.shape[:2]

            if h > raw_w:
                i = random.randint(0, h - raw_h)
                j = random.randint(0, w - raw_h)
                for k, v in sample.items():
                    sample[k] = v[i:i + raw_h, j:j + raw_h]

            else:
                res_h = raw_w - h
                for k, v in sample.items():
                    sample[k] = cv.copyMakeBorder(v, res_h, 0, res_h, 0, borderType=cv.BORDER_REFLECT)

            return sample
        else:
            return sample


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, sample):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            img = sample['image']
            height, width = img.shape[0:2]
            mat = cv.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            for k, v in sample.items():
                sample[k] = cv.warpAffine(v, mat, (height, width),
                                     flags=cv.INTER_LINEAR,
                                     borderMode=cv.BORDER_REFLECT_101)

        return sample


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):

        for t in self.transforms:
            sample = t(sample)
        return sample


class BIPEDDataset(Dataset):
    def __init__(self, img_root, mode='train', crop_size=448, base_size=512):
        self.img_root = img_root
        self.mode = mode
        self.imgList = os.listdir(img_root)
        self.crop_size = crop_size
        self.base_size = base_size
        self.transforms = DualCompose([
                # Rotate(),
                RandomFlip(),
                RandomRotate90(),
                # Rescale(scaleList[random.randint(0, len(scaleList) - 1)])
            ])

    def __len__(self):
        return len(self.imgList)

    def _make_img_gt_point_pair(self, index):
        imgPath = os.path.join(self.img_root, self.imgList[index])
        assert os.path.exists(imgPath), 'please check if the image path exists'
        file_name = os.path.basename(imgPath).split('.')[0]
        labelRoot = imgPath.replace('images', 'labels')
        _connect0_Root = imgPath.replace('images', 'connect_8_d1').split('.png')[0] + '_0.png'
        _connect1_Root = imgPath.replace('images', 'connect_8_d1').split('.png')[0] + '_1.png'
        _connect2_Root = imgPath.replace('images', 'connect_8_d1').split('.png')[0] + '_2.png'
        _connect_d1_0_Root = imgPath.replace('images', 'connect_8_d3').split('.png')[0] + '_0.png'
        _connect_d1_1_Root = imgPath.replace('images', 'connect_8_d3').split('.png')[0] + '_1.png'
        _connect_d1_2_Root = imgPath.replace('images', 'connect_8_d3').split('.png')[0] + '_2.png'

        _img = Image.open(imgPath).convert('RGB')
        _target = Image.open(labelRoot)
        _connect0 = Image.open(_connect0_Root).convert('RGB')
        _connect1 = Image.open(_connect1_Root).convert('RGB')
        _connect2 = Image.open(_connect2_Root).convert('RGB')
        _connect_d1_0 = Image.open(_connect_d1_0_Root).convert('RGB')
        _connect_d1_1 = Image.open(_connect_d1_1_Root).convert('RGB')
        _connect_d1_2 = Image.open(_connect_d1_2_Root).convert('RGB')

        sample = {'image': _img, 'label': _target, 'connect0': _connect0, 'connect1': _connect1, 'connect2': _connect2,
                  'connect_d1_0': _connect_d1_0, 'connect_d1_1': _connect_d1_1, 'connect_d1_2': _connect_d1_2}

        return sample, file_name

    def __getitem__(self, index):
        sample, file_name = self._make_img_gt_point_pair(index)

        if self.mode == "train":
            sample = self.transform_tr(sample)
        elif self.mode == 'val':
            sample = self.transform_val(sample)

        sample['file_name'] = file_name
        return sample

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomRotate(180),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

# class balance weight map
def balancewm(mask):
    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [1.0 / np.sum(mask==i) for i in classes ]
    freq /= max(freq)


    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]

    return wc


def distranfwm(mask, beta=3):
    mask = mask.astype('float')
    wc = balancewm(mask)

    dwm = ndimage.distance_transform_edt(mask != 1)
    dwm[dwm > beta] = beta
    dwm = wc + (1.0 - dwm / beta) + 1

    return dwm


if __name__ == '__main__':

    from config import Config
    cfg = Config()
    root = r'D:\2022\2\road_extraction\CoANet-main\data\deepglobe\crops\images'
    train_dataset = BIPEDDataset(root, crop_size=448, base_size=512)
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0)
    print(len(train_loader))

    for i, data_batch in enumerate(train_loader):
        # if i > 10:
        #     break

        img, dt = data_batch['image'], data_batch['label']
        connect0 = data_batch["connect0"]
        print(img.size(), connect0.size(), f"max gt : {torch.max(connect0)}")

        tmp = np.array(dt[0].numpy()).astype(np.uint8)
        img_tmp = np.transpose(img[0].numpy(), axes=[1, 2, 0])
        img_tmp *= (0.229, 0.224, 0.225)
        img_tmp += (0.485, 0.456, 0.406)
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        plt.figure()
        plt.title('display')
        plt.subplot(211)
        plt.imshow(img_tmp)
        plt.subplot(212)
        plt.imshow(tmp)
        plt.show()

    # crop_size = 400
    # scaleList = [int(crop_size * 0.75),
    #              int(crop_size * 0.875),
    #              crop_size,
    #              int(crop_size * 1.125)]
    # record = []
    # for i in range(1000):
    #     try:
    #         a0 = random.randint(0, len(scaleList))
    #         a = scaleList[a0]
    #     except:
    #         print(f'error {a0}')
    #     if a not in record:
    #         record.append(a)
    # print(record)

    # sample = {'a':1, 'b':2}
    #
    # for k in sample:
    #     camera_frame = sample[k]
