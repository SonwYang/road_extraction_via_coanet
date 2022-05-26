import cv2
import os
import glob
import gdalTools
import numpy as np

if __name__ == '__main__':
    gtList = glob.glob(r"D:\MyWorkSpace\dl_dataset\road_extraction\masa\test\labels\*.tif")
    outRoot = r"D:\MyWorkSpace\dl_dataset\road_extraction\masa\test\png"
    gdalTools.mkdir(outRoot)

    for imgPath in gtList:
        baseName = os.path.basename(imgPath).split(".")[0]
        outPath = os.path.join(outRoot, baseName + ".png")
        gt = cv2.imread(imgPath, 0)
        gt = np.where(gt > 0, 255, 0)
        cv2.imwrite(outPath, gt)