from torch.utils.data import Dataset
import numpy as np
from cv2 import *
from torchvision import transforms


def default_loader(path):
    # img = cv2.imread(path)
    img = imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    q_image = cv2.resize(img, dsize=(224, 224), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    # # 转为灰度图
    # q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2GRAY)
    # 数值两极化
    thresh, q_image = cv2.threshold(q_image, thresh=220, maxval=255, type=cv2.THRESH_BINARY)
    # 转为tensor格式
    img_tensor = transforms.ToTensor()(q_image)
    return img_tensor

class customData(Dataset):
    def __init__(self, images_path, labels, loader=default_loader):
        self.images = images_path
        self.labels = labels
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        img = self.loader(path)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    path = "E:/TEST/CASIA_data\\train\\一\\101625.png"
    img = imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # path_gbk = path.encode('gbk')
    # img = cv2.imread(path_gbk.decode())
    q_image = cv2.resize(img, dsize=(224, 224), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    # # 转为灰度图
    # q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2GRAY)
    # 数值两极化
    thresh, q_image = cv2.threshold(q_image, thresh=220, maxval=255, type=cv2.THRESH_BINARY)
    cv2.imshow("tex",q_image)
    cv2.waitKey(0)



