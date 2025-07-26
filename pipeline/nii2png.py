import os
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm
import pickle
import cv2
import matplotlib.pyplot as plt

'''
窗口化处理
'''
def windowing(img, window_width, window_center):
    # img:目标图片
    # window_width:窗宽
    # window_center:窗中心
    minWindow = float(window_center) - 0.5 * float(window_width)
    new_img = (img - minWindow) / float(window_width)
    new_img[new_img<0] = 0
    new_img[new_img>1] = 1
    return (new_img * 255).astype('uint8')

'''
直方图均衡化
'''
def clahe_equalized(imgs):
    # shape like (129, 512, 512)
    assert(len(imgs.shape) == 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_res = np.zeros_like(imgs)
    for i in range(len(imgs)):
        img_res[i, :, :] = clahe.apply(np.array(imgs[i, :, :], dtype='uint8'))
    return img_res / 255

'''
分为训练集和测试集
'''
def split_train_val(data_path='', percent=1.0):    # percent直接定义为1  表示不取测试集
    # percent：训练集比例
    # np.random.seed(0)
    origin_volume = [volume for volume in os.listdir(data_path) if volume.endswith('.nii')]
    # origin_volume = np.random.permutation(origin_volume)  #测试时不进行序列的随机排序
    train_volume = origin_volume[: int(percent * len(origin_volume))]
    val_volume = origin_volume[int(percent * len(origin_volume)) :]
    return train_volume, val_volume



def preprocessing_liver(volumes, zoom=True, save_path='',data_path='', tumor=False, pos=0):
    # os.makedirs(os.path.join(save_path, 'class'), exist_ok=True)
    for ii, volume in tqdm(enumerate(volumes), total=len(volumes)):
        index = 1
        ct = sitk.ReadImage(os.path.join(data_path, volume), sitk.sitkInt16) 
        ct_array = sitk.GetArrayFromImage(ct)  
        if pos == 0:
            ct_array = windowing(ct_array, 250, 45)     # 窗口化处理  图片分类先统一为(250, 45)
        if pos == 1:
            ct_array = windowing(ct_array, 20, 160)     # 窗口化处理   椎骨横突分类
        if pos == 2:
            ct_array = windowing(ct_array, 400, 0)
        # ct_array = clahe_equalized(ct_array)    # 直方图均衡化（没用）

        # 标签中像素
        if tumor == True:     # 图片分类不用管这部分代码
            seg_array[seg_array == 1] = 0
            seg_array[seg_array == 2] = 1
        # else:
        #     seg_array[seg_array == 2] = 1 # 暂不执行此操作

        # 考虑是否下采样，默认False
        if zoom == True:       #  图片分类不用管这部分行代码
            ct_array = ndimage.zoom(ct_array, 0.5, order=1)
            seg_array = ndimage.zoom(seg_array, 0.5, order=0)

        # 将上述切片取出，转化为png格式

        for i in range(ct_array.shape[0]):
            ct_image = ct_array[i, :, :]  # 载入的图像与正常图像是上下翻转的关系  (使用转化序列图片大于3的dcm2nii方法)   载入图像正常(用转化序列图片小于3的dcm2nii方法)
            ct_image = np.rot90(np.transpose(ct_image, (1, 0)))  #采用转置加旋转90的方法可实现翻转图像正常化
            volume_name = volume.replace('Volume_', '').replace('.nii', '')
            #保存切片为png格式到指定的文件夹中，命名为例：1_1_class.png
            
            plt.imsave(os.path.join(save_path, f'{volume_name}_{index}_class.png'),
                       ct_image, cmap='gray')     # Png图片保存在save/class里面
            # if index == 19:
            #     plt.imsave(os.path.join('/mnt/sda/yx/wzj/pipeline/demo/a', f'{volume_name}_{index}_class.png'),ct_image, cmap='gray')

            index += 1

def get_dicom(volumes, best, save_path='',data_path='', tumor=False, pos=0):
    for ii, volume in tqdm(enumerate(volumes), total=len(volumes)):
        index = 1
        ct = sitk.ReadImage(os.path.join(data_path, volume), sitk.sitkInt16) 
        ct_array = sitk.GetArrayFromImage(ct)
        for i in range(ct_array.shape[0]):
            index += 1
            if index == best:
                #找出nii文件对应的DICOM名称
                return 




def delete_unique_files_in_save_judge(l3_class_dir, save_judge_dir):
    """
    删除save_judge文件夹中不存在于L3_class文件夹中的同名文件
    
    参数:
        l3_class_dir: L3_class文件夹路径
        save_judge_dir: save_judge文件夹路径
    """
    # 获取两个文件夹中的文件名集合
    l3_files = set(os.listdir(l3_class_dir))
    save_judge_files = set(os.listdir(save_judge_dir))
    
    # 找出只在save_judge中存在的文件
    files_to_delete = save_judge_files - l3_files
    
    # 删除这些文件
    for filename in files_to_delete:
        file_path = os.path.join(save_judge_dir, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"删除失败 {file_path}: {str(e)}")
    

if __name__ == '__main__':
    data_path = '/mnt/sda/yx/wzj/pipeline/demo'
    save_path = os.path.join(data_path, 'save')  # 先组合路径
    os.makedirs(save_path, exist_ok=True) 
    # filp_list = ['Volume_262.nii', 'Volume_263.nii', 'Volume_264.nii', 'Volume_265.nii', 'Volume_266.nii', 'Volume_267.nii', 'Volume_268.nii', 'Volume_269.nii']
    # filp_list = ['volume_48.nii']
    volumes_train, volumes_val = split_train_val(data_path, percent=1.0)
    preprocessing_liver(volumes_train, False, save_path=save_path, tumor=False)
    # preprocessing_liver(filp_list, False, save_path=save_path, tumor=False)

