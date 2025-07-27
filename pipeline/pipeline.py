import os
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from networks.models import DenseNet121
from tqdm import tqdm
from PIL import Image
from nii2png import split_train_val,preprocessing_liver,delete_unique_files_in_save_judge
from infer_class import create_model,load_image_paths_from_folder,inference_on_images
from infer_judge import inference_on_images_judge
from utils.loss import *  
from utils.metrics import *
import bladder
import utils.image_transforms as joint_transforms
import utils.transforms as extended_transforms
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
import dicom2nifti  #转化序列数量大于3的
from compute import infer_density_area
import shutil
import SimpleITK as sitk

def segmentation(save_path):
   
    model_type = "ResU_MNetnew"  
    if model_type == "ResU_MNet"or model_type == "ResU_MNetnew":
        from networks.ResU_MNet import CLA_MNet

    root_path = './pipeline'
    model_path = os.path.join(root_path, 'model_path', 'ResU_MNetnew_depth=2_fold_1_dice_917116.pth')
    val_path = os.path.join(root_path, 'demo/save_seg')
    os.makedirs(save_path, exist_ok=True)
    input_transform = extended_transforms.NpyToTensor()
    target_transform = extended_transforms.MaskToTensor()
    palette = [[0], [85], [170], [255]]
    fold = 1

    val_set = bladder.Dataset(val_path, 'infer', fold,
                              joint_transform=None, transform=input_transform, center_crop=None,  
                              target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    net = CLA_MNet(1, classses=bladder.num_classes).cuda()  
    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    net.eval()
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
        cv2.imwrite(os.path.join(save_path, file_name[0][:]), np.uint8(pred))
        


def convert_all_dicom_to_nifti(data_ct_root, copy_target_dir):
    os.makedirs(copy_target_dir, exist_ok=True)

    for root, dirs, files in os.walk(data_ct_root):
        # 如果当前目录包含 DICOM 文件（.dcm），则进行转换
        dicom_files = [f for f in files if f.lower().endswith('.dcm')]
        # print(dicom_files)
        if dicom_files:
            output_path = os.path.join(root, 'Volume_1.nii')
            dicom2nifti.dicom_series_to_nifti(root, output_path, reorient_nifti=True)

            # 使用当前子目录名生成唯一目标文件名
            folder_name = os.path.basename(output_path)
            # print(folder_name)
            target_nii_path = os.path.join(copy_target_dir, f'{folder_name}')

            shutil.copy(output_path, target_nii_path)

   
if __name__ == '__main__':
    #数据预处理
    data_ct = './pipeline/ct'
    data_path = './pipeline/demo'
    # dicom2nifti.dicom_series_to_nifti(data_ct, os.path.join(data_ct, 'Volume_case.nii'), reorient_nifti=True)
    convert_all_dicom_to_nifti(data_ct, data_path)
    save_path_Class = os.path.join(data_path, 'save_class')
    save_path_Judge = os.path.join(data_path, 'save_judge')
    save_path_Seg = os.path.join(data_path, 'save_seg')
    os.makedirs(save_path_Class, exist_ok=True)
    os.makedirs(save_path_Judge, exist_ok=True)
    os.makedirs(save_path_Seg, exist_ok=True)
    volumes_train, volumes_val = split_train_val(data_path, percent=1.0)
    preprocessing_liver(volumes_train, False, save_path_Class, data_path, tumor=False, pos=0)
    preprocessing_liver(volumes_train, False, save_path_Judge, data_path, tumor=False, pos=1)
    preprocessing_liver(volumes_train, False, save_path_Seg, data_path, tumor=False, pos=2)
    
    #L3分类
    model_path = "./pipeline/model_path/best_model_class.pth"
    image_folder = "./pipeline/demo/save_class"  # 推理用图像目录（不含标签）
    save_txt_path = "inference_results_class.txt"
    save_l3_class_dir = "./pipeline/demo/L3_class"
    batch_size = 8  # 设置批量大小
    model = create_model(out_size=5)
    checkpoint = torch.load(model_path,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    image_paths = load_image_paths_from_folder(image_folder)
    results = inference_on_images(model, image_paths, 
                                  save_txt_path=save_txt_path, 
                                  save_l3_dir = save_l3_class_dir,
                                  target=2)

    #椎突判断
    delete_unique_files_in_save_judge(save_l3_class_dir, save_path_Judge)
    # save_path_Judge = "./pipeline/demo/L3_judge"
    model_path = "./pipeline/model_path/best_model_judge.pth"
    image_folder = "./pipeline/demo/save_judge"  # 推理用图像目录（不含标签）
    save_txt_path = "inference_results_judge.txt"
    save_l3_judge_dir = "./pipeline/demo/L3_Judge"
    batch_size = 8  # 设置批量大小
    model = create_model(out_size=3)
    checkpoint = torch.load(model_path,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    image_paths = load_image_paths_from_folder(image_folder)
    results = inference_on_images_judge(model, image_paths, 
                                  save_txt_path=save_txt_path,
                                  save_l3_dir =save_l3_judge_dir,
                                  target=0)
    if not results:
        print("由于CT间距较大，未能找到有椎突的L3切片")
    else:    
        #定位最佳L3
        # print("result",results)
        L3 = []
        L3_map = {}  # 用于记录：数字 → 原始文件名
        for i, (filename, label) in enumerate(results):
            if label == 0:
                # 提取中间的数字
                number = int(filename.split('_')[1])  # '1_25_class.png' → ['1', '25', 'class.png']
                L3.append(number)
                L3_map[number] = filename  # 记录映射
        # print(L3)
        files = sorted(os.listdir('/mnt/sda/yx/wzj/pipeline/ct/volume-64/ScalarVolume_526/'))
        # print(files)

        # 遍历 L3，从下往上选出第 i 个文件
        selected_files = []
        for i in L3:
            if i <= len(files):
                selected_file = files[-i]  # 从后往前第 i 个
                selected_files.append(selected_file)
            else:
                print(f"Warning: index {i} is out of range (only {len(files)} files).")

        # print("选中的文件：", selected_files)
        max_global_count = 0
        best_global_file = None
        for i, file in zip(L3, selected_files):
            nii_file_path = os.path.join('/mnt/sda/yx/wzj/pipeline/ct/volume-64/ScalarVolume_526/', file)
            
            img = sitk.ReadImage(nii_file_path)
            img_data = sitk.GetArrayFromImage(img)

            slice_data = img_data[0]  # 取出唯一一张切片
            # print('slice_data',img_data.shape,slice_data)
            count = np.sum(slice_data > 150)  # 统计 HU > 150 的像素数
            # print(f"文件: {file}, HU>150 像素数: {count}")

            if count > max_global_count:
                max_global_count = count
                best_global_file = file
                best_global_i = i  # 记录对应的 i

        # print("HU 值大于 150 像素最多的图像：")
        # print(f"文件: {best_global_file}, 对应 L3 中的索引: {best_global_i}")
        # print(f"原始标注文件名（来自 results）: {L3_map[best_global_i]}")
        # print(f"HU > 150 像素数: {max_global_count}")

        keep_filename = L3_map[best_global_i]
        # print(f"保留文件: {keep_filename}")

        # 遍历 save_path_Seg 下的所有文件
        for fname in os.listdir(save_path_Seg):
            fpath = os.path.join(save_path_Seg, fname)
            if os.path.isfile(fpath) and fname != keep_filename:
                os.remove(fpath)
            # elif fname == keep_filename:
            #     print(f"保留: {fname}")

        #分割
        # delete_unique_files_in_save_judge(save_l3_judge_dir, save_path_Seg)
        # save_path_pre = '/mnt/sda/yx/wzj/pipeline/demo/L3_seg'
        save_path_pre = os.path.join(data_path, 'seg_pre')
        segmentation(save_path_pre)

        #计算
        # delete_unique_files_in_save_judge(save_path_Seg, save_path_Compute)
        infer_density_area(data_ct, save_path_Seg, save_path_pre, img_data, file_number=0)
        infer_density_area(data_ct, save_path_Seg, save_path_pre, img_data, file_number=1)
        infer_density_area(data_ct, save_path_Seg, save_path_pre, img_data, file_number=2)
    



    
