import argparse
import os
import numpy as np
import time
from modeling.coanet import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid #, save_image
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize([1300, 1300])
    im.save(filename)


def main():
    parser = argparse.ArgumentParser(description="PyTorch CoANet Training")
    parser.add_argument('--out-path', type=str, default='./run/spacenet/CoANet-resnet',
                        help='mask image to save')
    parser.add_argument('--img_path', type=str, default=r'D:\MyWorkSpace\MyUtilsCode\road_extraction\COANet\data\crop\images\10828795_15_3.png',
                        help='path to image')
    parser.add_argument('--backbone', type=str, default='resnet',
                        help='backbone name (default: resnet)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for test ')
    parser.add_argument('--ckpt', type=str, default=r'D:\MyWorkSpace\MyUtilsCode\road_extraction\COANet\epoch0023_model.pth',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--loss-type', type=str, default='con_ce',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='DeepGlobe',
                        choices=['spacenet', 'DeepGlobe'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size. spacenet:1280, DeepGlobe:1024.')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size. spacenet:1280, DeepGlobe:1024.')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    kwargs = {'num_workers': args.workers, 'pin_memory': False}
    device = 'cuda:0'

    model = torch.load(args.ckpt, map_location=device)


    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # out_path = os.path.join(args.out_path, 'out_imgs_1300/')
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)

    model.eval()

    img = cv2.imread(r"D:\MyWorkSpace\MyUtilsCode\road_extraction\COANet\data\crop\images\10828795_15_3.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    image = composed_transforms(image)
    image = torch.from_numpy(np.array([image.numpy()]))
    image = image.cpu().numpy()
    image1 = image[:, :, ::-1, :]
    image2 = image[:, :, :, ::-1]
    image3 = image[:, :, ::-1, ::-1]
    image = np.concatenate((image,image1,image2,image3), axis=0)
    image = torch.from_numpy(image).float()

    if args.cuda:
        image = image.cuda()
    with torch.no_grad():
        output, out_connect, out_connect_d1 = model(image)

    out_connect_full = []
    out_connect = out_connect.data.cpu().numpy()
    out_connect_full.append(out_connect[0, ...])
    out_connect_full.append(out_connect[1, :, ::-1, :])
    out_connect_full.append(out_connect[2, :, :, ::-1])
    out_connect_full.append(out_connect[3, :, ::-1, ::-1])
    out_connect_full = np.asarray(out_connect_full).mean(axis=0)[np.newaxis, :, :, :]
    pred_connect = np.sum(out_connect_full, axis=1)
    dn = np.squeeze(pred_connect.copy())
    dn = (dn - np.min(dn)) / np.max(dn) * 255
    io.imsave("pred_connect.png", dn.astype(np.uint8))

    pred_connect[pred_connect < 0.5] = 0
    pred_connect[pred_connect >= 0.5] = 1

    out_connect_d1_full = []
    out_connect_d1 = out_connect_d1.data.cpu().numpy()
    out_connect_d1_full.append(out_connect_d1[0, ...])
    out_connect_d1_full.append(out_connect_d1[1, :, ::-1, :])
    out_connect_d1_full.append(out_connect_d1[2, :, :, ::-1])
    out_connect_d1_full.append(out_connect_d1[3, :, ::-1, ::-1])
    out_connect_d1_full = np.asarray(out_connect_d1_full).mean(axis=0)[np.newaxis, :, :, :]
    pred_connect_d1 = np.sum(out_connect_d1_full, axis=1)

    dn = np.squeeze(pred_connect_d1.copy())
    dn = (dn - np.min(dn)) / np.max(dn) * 255
    io.imsave("pred_connect_d1.png", dn.astype(np.uint8))

    pred_connect_d1[pred_connect_d1 < 1.0] = 0
    pred_connect_d1[pred_connect_d1 >= 1.0] = 1

    pred_full = []
    pred = output.data.cpu().numpy()
    pred_full.append(pred[0, ...])
    pred_full.append(pred[1, :, ::-1, :])
    pred_full.append(pred[2, :, :, ::-1])
    pred_full.append(pred[3, :, ::-1, ::-1])
    pred_full = np.asarray(pred_full).mean(axis=0)

    dn = np.squeeze(pred_full.copy())
    dn = (dn - np.min(dn)) / np.max(dn) * 255
    io.imsave("pred_full.png", dn.astype(np.uint8))

    pred_full[pred_full > 0.05] = 1
    pred_full[pred_full < 0.05] = 0

    su = pred_full + pred_connect + pred_connect_d1
    # dn = np.squeeze(su.copy())
    # dn = (dn - np.min(dn)) / np.max(dn) * 255
    # io.imsave("dn.png", dn.astype(np.uint8))
    su[su > 0] = 1
    su = np.squeeze(su.astype(int))

    plt.subplot(121)
    plt.title("image")
    plt.imshow(img)
    plt.subplot(122)
    plt.title("prediction")
    plt.imshow(su)
    plt.show()

    # print(f"su shape : {su.shape}")
    # io.imsave("result.png", su)


if __name__ == "__main__":
   main()


## python test.py --ckpt='./run/DeepGlobe/CoANet-resnet/CoANet-DeepGlobe.pth.tar' --out_path='./run/DeepGlobe/CoANet-resnet' --dataset='DeepGlobe' --base_size=1024 --crop_size=1024
