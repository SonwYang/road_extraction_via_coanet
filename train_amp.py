from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import GradScaler, autocast
from datasetGeo import BIPEDDataset
from utils.loss import *
from config import Config
import segmentation_models_pytorch as smp
from cyclicLR import CyclicCosAnnealingLR, LearningRateWarmUP
from modeling.coanet import *
import torchgeometry as tgm
import numpy as np
import time
import os
import cv2 as cv
from tqdm import tqdm
import glob
from random import sample
from lookahead import Lookahead
from utils.metrics import Evaluator
import warnings
warnings.filterwarnings("ignore")


def weight_init(m):
    if isinstance(m, (nn.Conv2d, )):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)
        if m.weight.data.shape == torch.Size([1,6,1,1]):
            torch.nn.init.constant_(m.weight,0.2)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):

        torch.nn.init.normal_(m.weight,mean=0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.model = FishUnet(num_classes=2, in_channels=8, encoder_depth=34).to(self.device).apply(weight_init)
        self.model = CoANet(num_classes=cfg.num_classes,
                            backbone=cfg.backbone,
                            output_stride=cfg.out_stride,
                            sync_bn=cfg.sync_bn,
                            freeze_bn=cfg.freeze_bn).to(self.device)
        #self.model = HighResolutionNet(num_classes=3, in_chs=8).to(self.device)
        self.criterion = dice_bce_loss()
        self.criterion_con = SegmentationLosses(weight=None, cuda=self.device).build_loss(mode="con_ce")

        # Define Evaluator
        self.evaluator = Evaluator(2)

        optimizer = torch.optim.AdamW([
                {'params': self.model.parameters()},
                # {'params': self.awl.parameters(), 'weight_decay': 0}
            ])
        self.optimizer = Lookahead(optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, verbose=True)
        self.scheduler = LearningRateWarmUP(optimizer=optimizer, target_iteration=10, target_lr=0.0005,
                                            after_scheduler=scheduler)
        self.scaler = GradScaler()
        mkdir(cfg.model_output)

    def load_net(self, resume):
        self.model = torch.load(resume,  map_location=self.device)
        print('load pre-trained model successfully')

    def build_loader(self):
        imglist = glob.glob(f'{self.cfg.train_root}/*')[:10000]
        indices = list(range(len(imglist)))
        indices = sample(indices, len(indices))
        split = int(np.floor(0.15 * len(imglist)))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        print(f'Total images {len(imglist)}')
        print(f'Number of train images {len(train_idx)}')
        print(f'Number of validation images {len(valid_idx)}')

        train_dataset = BIPEDDataset(self.cfg.train_root,
                                     crop_size=self.cfg.crop_size,
                                     base_size=self.cfg.base_size)
        valid_dataset = BIPEDDataset(self.cfg.train_root,
                                     crop_size=self.cfg.crop_size,
                                     base_size=self.cfg.base_size)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.cfg.batch_size,
                                  num_workers=self.cfg.num_workers,
                                  shuffle=False,
                                  sampler=train_sampler,
                                  drop_last=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.cfg.batch_size,
                                  num_workers=self.cfg.num_workers,
                                  shuffle=False,
                                  sampler=valid_sampler,
                                  drop_last=True)
        return train_loader, valid_loader

    def validation(self, epoch, dataloader):
        self.model.eval()
        self.evaluator.reset()
        running_loss = []
        for batch_id, sample in enumerate(dataloader):
            image, target, con0, con1, con2, con_d1_0, con_d1_1, con_d1_2 = \
                sample['image'], sample['label'], sample['connect0'], sample['connect1'], sample['connect2'], \
                sample['connect_d1_0'], sample['connect_d1_1'], sample['connect_d1_2']
            connect_label = torch.cat((con0, con1, con2), 1)
            connect_d1_label = torch.cat((con_d1_0, con_d1_1, con_d1_2), 1)

            image, target, connect_label, connect_d1_label = image.to(self.device), target.to(
                self.device), connect_label.to(self.device), connect_d1_label.to(self.device)
            with autocast():
                output, out_connect, out_connect_d1 = self.model(image)

                target = torch.unsqueeze(target, 1)
                loss1 = self.criterion(output, target)
                loss2 = self.criterion_con(out_connect, connect_label)
                loss3 = self.criterion_con(out_connect_d1, connect_d1_label)
                lad = 0.2
                loss = loss1 + lad * (0.6 * loss2 + 0.4 * loss3)

            print(time.ctime(), 'validation, Epoch: {0} Sample {1}/{2} Loss: {3}' \
                  .format(epoch, batch_id, len(dataloader), loss.item()), end='\r')
            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            # Add batch sample into evaluator
            pred[pred > 0.1] = 1
            pred[pred < 0.1] = 0
            self.evaluator.add_batch(target_n, pred)
            self.save_image_bacth_to_disk(output, sample["file_name"])
            running_loss.append(loss.detach().item())
            return np.mean(np.array(running_loss)), self.evaluator.Intersection_over_Union()

    def save_image_bacth_to_disk(self, tensor, file_names):
        output_dir = self.cfg.valid_output_dir
        mkdir(output_dir)
        assert len(tensor.shape) == 4, tensor.shape
        for tensor_image, file_name in zip(tensor, file_names):
            image_vis = tgm.utils.tensor_to_image(torch.sigmoid(tensor_image))
            image_vis = (255.0 * (1.0 - image_vis)).astype(np.uint8)  #
            output_file_name = os.path.join(output_dir, f"{file_name}.png")
            cv.imwrite(output_file_name, image_vis)

    def train(self):
        train_loader, valid_loader = self.build_loader()
        best_loss = 1000000
        best_iou = 0
        best_train_loss = 1000000
        valid_losses = []
        train_losses = []

        for epoch in range(1, self.cfg.num_epochs):
            self.model.train()
            running_loss = []
            for batch_id, sample in enumerate(train_loader):
                image, target, con0, con1, con2, con_d1_0, con_d1_1, con_d1_2 = \
                    sample['image'], sample['label'], sample['connect0'], sample['connect1'], sample['connect2'], \
                    sample['connect_d1_0'], sample['connect_d1_1'], sample['connect_d1_2']
                connect_label = torch.cat((con0, con1, con2), 1)
                connect_d1_label = torch.cat((con_d1_0, con_d1_1, con_d1_2), 1)

                image, target, connect_label, connect_d1_label = image.to(self.device), target.to(self.device), connect_label.to(self.device), connect_d1_label.to(self.device)
                with autocast():
                    output, out_connect, out_connect_d1 = self.model(image)

                    target = torch.unsqueeze(target, 1)
                    loss1 = self.criterion(output, target)
                    loss2 = self.criterion_con(out_connect, connect_label)
                    loss3 = self.criterion_con(out_connect_d1, connect_d1_label)
                    lad = 0.2
                    loss = loss1 + lad * (0.6 * loss2 + 0.4 * loss3)
                self.scaler.scale(loss).backward()
                # 先反缩放梯度，若反缩后梯度不是inf或者nan，则用于权重更新
                self.scaler.step(self.optimizer)
                # 更新缩放器
                self.scaler.update()


                print(time.ctime(), 'training, Epoch: {0} Sample {1}/{2} Loss: {3}'\
                      .format(epoch, batch_id, len(train_loader), loss.item()), end='\r')

                running_loss.append(loss.detach().item())

            train_loss = np.mean(np.array(running_loss))

            valid_loss, val_iou = self.validation(epoch, valid_loader)

            if epoch > 10:
                self.scheduler.after_scheduler.step(valid_loss)
            else:
                self.scheduler.step(epoch)

            lr = float(self.scheduler.after_scheduler.optimizer.param_groups[0]['lr'])

            if val_iou > best_iou:
                torch.save(self.model, os.path.join(self.cfg.model_output, f'epoch{str(epoch).zfill(4)}_model.pth'))
                modelList = glob.glob(os.path.join(self.cfg.model_output, f'epoch*_model.pth'))
                if len(modelList) > 3:
                    modelList = modelList[:-3]
                    for modelPath in modelList:
                        os.remove(modelPath)

                print(f'find optimal model, IOU {best_iou}==>{val_iou} \n')
                best_iou = val_iou
                # print(f'lr {lr:.8f} \n')
                valid_losses.append([valid_loss, lr])
                np.savetxt(os.path.join(self.cfg.model_output, 'valid_loss.txt'), valid_losses, fmt='%.6f')

            if train_loss < best_train_loss:
                torch.save(self.model, os.path.join(self.cfg.model_output, f'train_best_model.pth'))
                best_train_loss = train_loss
                train_losses.append([train_loss, lr])
                np.savetxt(os.path.join(self.cfg.model_output, 'train_loss.txt'), train_losses, fmt='%.6f')

            torch.save(self.model, os.path.join(self.cfg.model_output, f'last_model.pth'))

        # plt.ioff()
        # plt.show()


if __name__ == '__main__':
    import argparse
    from preprocessData import prepareData
    parser = argparse.ArgumentParser(
        description='''This is a code for training model.''')
    parser.add_argument('--dataRoot', type=str, default=r'D:\BaiduNetdiskDownload\data_3groups\GF3_Yangzhitang_Samples_Feature_sub5', help='path to the root of data')
    parser.add_argument('--in_chs', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=1, help='the number of class')
    args = parser.parse_args()

    print('The training dataset is preparing... Please wait!')
    # prepareData(args.dataRoot)
    config = Config()
    # config.in_chs = args.in_chs
    # config.num_classes = args.num_classes

    print("Everything is ok! It's time for training.")
    trainer = Trainer(config)
    trainer.train()





