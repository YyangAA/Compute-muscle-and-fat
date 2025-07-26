import os
import torch
# torch.hub.set_dir('./torch_hub') 
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from networks.models import DenseNet121
from tqdm import tqdm
from PIL import Image
# ============ Step 1: 加载模型 ============
def create_model(ema=False,out_size=5):
        # Network definition
        net = DenseNet121(out_size=out_size, mode='U-Ones', drop_rate=0.2)
        if len('0'.split(',')) > 1:
            net = torch.nn.DataParallel(net)  
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

# ============ Step 2: 加载图像列表 ============
def load_image_paths_from_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, filename))
    return sorted(image_paths)

# ============ Step 3: 推理流程 ============
def inference_on_images_judge(model, image_paths, save_txt_path=None, save_l3_dir="",target=2):
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    os.makedirs(save_l3_dir, exist_ok=True)
    results = []
    for img_path in tqdm(image_paths, desc="Inferencing"):
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).cuda()

        with torch.no_grad():
            _, output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            # 保存预测结果为2的图像
            if pred == target:
                save_path = os.path.join(save_l3_dir, os.path.basename(img_path))
                image.save(save_path)
                # print(f"Saved L3 image: {save_path}")
            # pred_label = cell_data.CLASS_NAMES[pred]
            results.append((os.path.basename(img_path), pred))

    # 可选：保存预测结果
    # if save_txt_path:
    #     with open(save_txt_path, "w") as f:
    #         for fname, label in results:
    #             f.write(f"{fname},{label}\n")
    #     print(f"预测结果已保存到 {save_txt_path}")

    return results

# ============ 主程序 ============
if __name__ == "__main__":
    model_path = "./pipeline/model_path/best_model_l3.pth"
    image_folder = "./pipeline/demo/save/class"  # 推理用图像目录（不含标签）
    save_txt_path = "inference_results1.txt"
    batch_size = 8  # 设置批量大小
    model = create_model()
    checkpoint = torch.load(model_path,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    image_paths = load_image_paths_from_folder(image_folder)
    results = inference_on_images_judge(model, image_paths, save_txt_path=save_txt_path)

    # 打印部分结果
    print("\n部分预测结果：")
    for fname, label in results[:10]:
        print(f"{fname} => {label}")
