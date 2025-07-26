import time
import os
import cv2
import torch
import random
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import sys
import numpy as np


import bladder
import utils.image_transforms as joint_transforms
import utils.transforms as extended_transforms
from utils.loss import *   # *表示加载相应代码库的所有类和函数
from utils.metrics import *


from utils import misc
from utils.pytorchtools import EarlyStopping
from utils.LRScheduler import PolyLR


if __name__ == "__main__":
    model_type = "ResU_MNetnew"  
    if model_type == "ResU_MNet"or model_type == "ResU_MNetnew":
        from networks.ResU_MNet import CLA_MNet


    root_path = './pipeline'
    input_transform = extended_transforms.NpyToTensor()
    target_transform = extended_transforms.MaskToTensor()
    palette = [[0], [85], [170], [255]]
    depth = 2
    fold = 1
    val_path = os.path.join(root_path, 'demo/L3_Judge')

    val_set = bladder.Dataset(val_path, 'infer', fold,
                              joint_transform=None, transform=input_transform, center_crop=None,  
                              target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    net = CLA_MNet(1, classses=bladder.num_classes).cuda()  
    net.load_state_dict(torch.load(os.path.join(root_path, 'model_path/ResU_MNetnew_depth=2_fold_1_dice_917116.pth'), 
                                                map_location='cuda'))
    net.eval()
    save_path = '/mnt/sda/yx/wzj/pipeline/demo/pred'
    os.makedirs(save_path, exist_ok=True)
    for val_batch, ((input, _), file_name) in tqdm(enumerate(val_loader, 1)):
        val_X = input.cuda()
        pred = net(val_X)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()
        # 输出预测图像代码
        pred = np.array(pred.data.cpu()[0])
        pred = helpers.onehot_to_mask(np.array(pred.squeeze()).transpose(1, 2, 0), palette)
        pred = helpers.array_to_img(pred)
        cv2.imwrite(os.path.join(save_path, file_name[0][-20:]), np.uint8(pred))
        




   