import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
from utils import helpers

'''
128 = bladder
255 = tumor
0   = background 
'''
palette = [[0], [85], [170], [255]]  # one-hot的颜色表     #  改动  85 肌肉  170 皮下脂肪  255 内脏脂肪  有疑问
num_classes = 4  # 分类数                     #   改动


def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test', 'infer']
    items = []
    if mode == 'train':
        # img_path = os.path.join(root, 'three_density_ct')   # 改动
        img_path = os.path.join(root, 'old_three_density_ct')   # 改动
        # mask_path = os.path.join(root, 'old_seg_three')  # 改动
        mask_path = os.path.join(root, 'seg_three')         # 改动

        if 'Augdata' in root:  # 当使用增广后的训练集
            data_list = os.listdir(os.path.join(root, 'Labels'))
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it.replace('three_density', 'seg_three')))    # 改动
            items.append(item)
    elif mode == 'val':
        # img_path = os.path.join(root, 'three_density_ct')   # 改动
        img_path = os.path.join(root, 'old_three_density_ct')
        mask_path = os.path.join(root, 'seg_three')    # 改动
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it.replace('three_density', 'seg_three')))    # 改动
            items.append(item)
    elif mode =='infer':
        img_path = os.path.join(root)  
        mask_path = None
        data_list = [f for f in os.listdir(img_path) if f.lower().endswith('.png')]
        for it in data_list:
            item = (os.path.join(img_path, it), None)  # 推理模式没有mask
            items.append(item)
        
    else:
        # img_path = os.path.join(root, 'Images')
        # data_list = [l.strip('\n') for l in open(os.path.join(
        #     root, 'test.txt')).readlines()]
        # for it in data_list:
        #     item = (os.path.join(img_path, 'c0', it))
        #     items.append(item)
        img_path = os.path.join(root, 'off_set_three_density_ct')  # 改动
        mask_path = os.path.join(root, 'off_set_seg_three')  # 改动
        # mask_path = os.path.join(root, 'old_seg_three')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'off_test{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it.replace('three_density', 'seg_three')))  # 改动
            items.append(item)
    return items


class Dataset(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)
        self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        file_name = img_path.split('/')[-1]

        img = Image.open(img_path)   # 并非按灰度图像读取,要重新转换为灰度图像 下同
        # mask = Image.open(mask_path)
        img = img.convert('L')
        # mask = mask.convert('L')

        if self.joint_transform is not None:
            img, _ = self.joint_transform(img)
        if self.center_crop is not None:
            img, _ = self.center_crop(img)
        img = np.array(img)
        # mask = np.array(mask)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        img = np.expand_dims(img, axis=2)
        # mask = np.expand_dims(mask, axis=2)
        # mask = helpers.mask_to_onehot(mask, self.palette)
        # shape from (H, W, C) to (C, H, W)
        img = img.transpose([2, 0, 1])
        # mask = mask.transpose([2, 0, 1])
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     mask = self.target_transform(mask)
        return (img, img), file_name



    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    from torch.utils.data import DataLoader
    import utils.image_transforms as joint_transforms
    import utils.transforms as extended_transforms

    # 测试加载数据类
    def demo():
        train_path = r'../media/Datasets/Bladder/raw_data'
        val_path = r'../media/Datasets/Bladder/raw_data'
        test_path = r'../media/Datasets/Bladder/test'

        center_crop = joint_transforms.CenterCrop(256)
        test_center_crop = joint_transforms.SingleCenterCrop(256)
        train_input_transform = extended_transforms.NpyToTensor()
        target_transform = extended_transforms.MaskToTensor()

        train_set = Dataset(train_path, 'train', 1,
                              joint_transform=None, center_crop=None,                    # 改动
                              transform=train_input_transform, target_transform=target_transform)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

        for (input, mask), file_name in train_loader:
            print(input.shape)
            print(mask.shape)
            img = helpers.array_to_img(np.expand_dims(input.squeeze(), 2))
            # squeeze() 是 NumPy 库中的一个函数，用于从数组的形状中移除长度为1的维度。axis: 可选参数，指定要移除的轴。默认情况下，移除所
            # 有长度为1的轴。如果指定轴，则只移除该轴，并且该轴必须是长度为1。
            # np.expand_dims 是 NumPy 库中的一个函数，用于在指定位置插入一个新的轴（维度）到数组中。这个新轴的长度为 1。
            # 将gt反one-hot回去以便进行可视化
            palette = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0]]          # 考虑改动
            # print(mask.squeeze().shape)
            # mask经过one-hot，通道数从1变为了K（分类类别数），再加上batch_size形状变为（1,K,h,w)
            # helpers.onehot_to_mask将图片从（h,w,k)变为（h,w,c)
            gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose(1, 2, 0), palette)
            gt = helpers.array_to_img(gt)
            # print(str(file_name[0]))
            # break
            # cv2.imshow('img GT', np.uint8(np.hstack([img, gt])))
            cv2.imwrite(os.path.join(train_path, 'save_img', str(file_name[0])), np.uint8(gt))
            # cv2.imshow('img GT', np.uint8(gt))
            # cv2.waitKey(1000)


    demo()
