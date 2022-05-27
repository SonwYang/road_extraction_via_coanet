# 1. 数据准备

## 1.1 原始数据格式

标签为二值图，后缀为png

```
- ${your data root}
	-- train_images
		--- xxx.tif
		--- xxx.tif
		...
	-- train_labels
		--- xxx.png
		--- xxx.png
		...
```

## 1.2 样本裁剪

```
cd ${your project root}/dataGenerators
python cropUtil.py --dataRoot ${your data root} --outRoot ${your data output root} --targteSize 512 --PaddingSize 128
```


## 1.3 生成connect标签

```
cd ${your project root}/dataGenerators
python create_connection.py --base_dir D:\MyWorkSpace\dl_dataset\road_extraction\masa\test\png
```

运行完成后，你的数据根目录如下

```
- ${your data root}
	-- train_images
		--- xxx.tif
		--- xxx.tif
		...
	-- train_labels
		--- xxx.png
		--- xxx.png
		...
	-- train_connect_8_d1
		--- xxx.png
		--- xxx.png
		...
	-- train_connect_8_d3
		--- xxx.png
		--- xxx.png
		...
```

# 2 模型训练

## 2.1 修改配置文件

```
class Config(object):
    #dataset
    crop_size = 512
    base_size = 640
    train_root = ${your data output root}/train_images
    valid_output_dir = 'valid_temp'
    resume = 'model.pth'

    # loss settings
    weight = False

    # hyper parameters
    batch_size = 2
    num_workers = 0
    num_epochs = 300
    model_output = 'ckpts_coanet'
    in_chs = 8
    ### model parameters
    num_classes = 1
    backbone = "resnet50"
    out_stride = 8
    sync_bn = False
    freeze_bn = False

```

## 2.2 运行训练脚本

`python train.py`



# 3 模型预测

`python predict_single.py --img_path 10828795_15_3.png --ckptepoch0023_model.pth `



