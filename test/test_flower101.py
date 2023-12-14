import os

# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import logging
import time
# import cv2
# import torch
# 伪代码示例
# 定义数据转换
# transform = transforms.Compose([transforms.Resize((256, 256)),
                                # transforms.ToTensor()])
logging.basicConfig(filename='xxn_performance_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# data_dir = '/data/ssd1/xuxinan/flower102jpg'
data_dir = '/data/ssd1/by/folowler'

# Get a list of file names in the directory
file_names = os.listdir(data_dir)

# Create the full file paths by joining the directory path with each file name
file_paths = [os.path.join(data_dir, file_name) for file_name in file_names]
# import torch

import time
class ImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        # self.image = torch.Tensor(512, 512, 33)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]

        # t1 = time.time()
        # print(img_path)
        image = Image.open(img_path).convert('RGB')
        # img = cv2.imread(img_path)
        # t2 = time.time()
        # print(t2 - t1)

        # image = cv2.imread(img_path)
        # image = cv2.resize(image, (512, 512))
        # if self.transform:
            # image = self.transform(image)
        # image = np.zeros()
        # time.sleep(0.001)

        return 1

# dataset = ImageDataset(file_paths=file_paths, transform=transform)
dataset = ImageDataset(file_paths=file_paths)

# 创建DataLoader并测试不同num_workers设置
# num_workers_list = [0, 4]
num_workers_list = [0, 4]

for num_workers in num_workers_list:

# num_workers = 4
# print(num_workers)
    logging.info(num_workers)
    start_time = time.time()
    # data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    # data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    # i=0

    # with torch.autograd.profiler.emit_itt() as prof:

    for batch in data_loader:
        pass
    # for i in dataset:
            # 在这里执行你的操作，例如模型推理
            # 这里仅仅是为了模拟一下数据加载后的处理时间
            # time.sleep(0.1)
            # print(i)
            # i+=1
            # print((time.time() - start_time) / i)
    end_time = time.time()
    print(f"num_workers={num_workers}: Performance metrics: "+str(end_time-start_time))
    logging.info(end_time-start_time)
    # print(prof.print_stats())
# 如果有性能提升，继续下一步；否则，可能是机器问题或其他原因，需要进一步调查。
