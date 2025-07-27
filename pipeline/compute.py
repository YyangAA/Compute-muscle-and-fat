import glob
import numpy as np
import torch
import os
import cv2
import pydicom
import csv
import fnmatch


def density_calculation(pred, img, trl):
    if trl == 30:
        left_bd, right_bd = -29, 150
    elif trl == 191:
        left_bd, right_bd = -190, -30
    elif trl == 151:
        left_bd, right_bd = -150, 150
    matting = pred * img
    matting[matting < left_bd] = 0
    matting[matting > right_bd] = 0
    # print("matting shape",matting.shape)
    # print(matting.max())
    # if trl == 151:
    #     image = matting.squeeze(0).squeeze(0)  # 形状变为 [512, 512]
    #     # 2. 转换为 NumPy 数组
    #     image_np = image.cpu().numpy()
    #     # 3. 缩放值域到 [0, 255] 并转换为 uint8
    #     image_np = (image_np).astype(np.uint8)
    #     cv2.imwrite('matting.png',image_np)
    total_hu = torch.sum(matting)
    total_pixel = torch.sum(pred)
    # print((total_hu.float() / total_pixel.float()))
    return (total_hu.float() / total_pixel.float())


def area_calculation(pred, horizontal_spacing, vertical_spacing):
    total_pixel = torch.sum(pred)
    return total_pixel.numpy() * (horizontal_spacing * vertical_spacing)


def save_to_csv(data, filename):
    keys = ['id', 'name', 'time', 'density(HU)', 'area(CM^2)']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def infer_density_area(dicom_dir, test_img_dir, pred_mask_dir, image_array, file_number=2):#（400，0）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置区域对应的类别标签值和偏移量
    label_map = {
        0: {'label_val': 85,  'trl': 30,  'csv_name': 'muscle_density.csv'},
        1: {'label_val': 170, 'trl': 191, 'csv_name': 'subcutaneous_density.csv'},
        2: {'label_val': 255, 'trl': 151, 'csv_name': 'visceral_density.csv'},
    }

    label_val = label_map[file_number]['label_val']
    trl = label_map[file_number]['trl']
    csv_output_path = label_map[file_number]['csv_name']


    tests_path = glob.glob(os.path.join(test_img_dir, '*.png'))
    tests_path.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

    # 读取 spacing 信息
    spacing_info = []
    for patient in os.listdir(dicom_dir):
        patient_path = os.path.join(dicom_dir, patient)
        for time_folder in os.listdir(patient_path):
            dcm_path = os.path.join(patient_path, time_folder)
            img_list = os.listdir(dcm_path)
            volume_name = next((f for f in img_list if fnmatch.fnmatch(f, 'Volume_*.nii')), None)
            if not volume_name:
                continue
            dcm_file = glob.glob(os.path.join(dcm_path, '*.dcm'))[0]
            dcm_data = pydicom.dcmread(dcm_file)
            spacing_info.append({
                'id': volume_name,
                'time': time_folder,
                'name': patient,
                'spacex': float(dcm_data.PixelSpacing[0]),
                'spacey': float(dcm_data.PixelSpacing[1])
            })

    results = []

    # img_tensor = torch.from_numpy(image_array).unsqueeze(0).float()  # 转为 [1, 1, 512, 512]
    # print("img", img_tensor.max(), img_tensor.min())
    for img_path in tests_path:
        file_id = os.path.basename(img_path).split('.')[0]
        filename = os.path.basename(img_path)
        new_filename = filename.replace("fat_density", "seg_three")
        volume_id = 'Volume_' + file_id.split('_')[0] + '.nii'

        # 原图和预测mask
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
        # print("img",img_tensor.max(),img_tensor.min())
        # img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

        mask_path = os.path.join(pred_mask_dir, new_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[跳过] 未找到预测掩码：{mask_path}")
            continue
        mask_binary = (mask == label_val).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0)

        # 推理计算
        density = density_calculation(mask_tensor, img_tensor, trl).item()

        match = next((s for s in spacing_info if s['id'] == volume_id), None)
        if not match:
            print(f"[跳过] 未找到 spacing 信息: {volume_id}")
            continue
        area = area_calculation(mask_tensor, match['spacex'], match['spacey'])/100

        # 记录并打印
        result = {
            'id': file_id,
            'name': match['name'],
            'time': match['time'],
            'density(HU)': round(density, 2),
            'area(CM^2)': round(area, 2)
            }
        print(f"[结果] {result}")
        results.append(result)

    save_to_csv(results, csv_output_path)
    print(f"推理完成，CSV结果保存在：{csv_output_path}")


# 运行
if __name__ == "__main__":
    infer_density_area(file_number=2)  # 修改为0/1/2根据区域选择
