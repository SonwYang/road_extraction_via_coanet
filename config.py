class Config(object):
    #dataset
    crop_size = 512
    base_size = 640
    train_root = r'D:\MyWorkSpace\dl_dataset\road_extraction\masa\test\train_images'
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
