import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Classify
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/classify')

class Args:
    def __init__(self) -> None:
        self.batch_size = 32
        self.lr = 0.001
        self.epochs = 1000
        self.data_len = 1000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = Args()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(c1=3,c2=4,k=3,s=2),
            Conv(c1=4,c2=8,k=3,s=2),
            C2f(c1=8,c2=8,shortcut=True),C2f(c1=8,c2=8,shortcut=True),C2f(c1=8,c2=8,shortcut=True),
            Conv(c1=8,c2=16,k=3,s=2),
            C2f(c1=16,c2=16,shortcut=True),C2f(c1=16,c2=16,shortcut=True),C2f(c1=16,c2=16,shortcut=True),
            C2f(c1=16,c2=16,shortcut=True),C2f(c1=16,c2=16,shortcut=True),C2f(c1=16,c2=16,shortcut=True),
            Conv(c1=16,c2=32,k=3,s=2),
            C2f(c1=32,c2=32,shortcut=True),C2f(c1=32,c2=32,shortcut=True),C2f(c1=32,c2=32,shortcut=True),
            C2f(c1=32,c2=32,shortcut=True),C2f(c1=32,c2=32,shortcut=True),C2f(c1=32,c2=32,shortcut=True),
            SPPF(c1=32,c2=32,k=5),
        )
        self.head = Classify(c1=32,c2=3)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def get_data():
    from generate import generate_img
    img1, text1 = generate_img()
    img2, text2 = generate_img()
    def normalize(image):
        mean = np.mean(image)
        var = np.mean(np.square(image-mean))
        image = (image - mean)/np.sqrt(var)
        return image
    img1 = normalize(img1)
    img2 = normalize(img2)
    img1 = np.array(img1,dtype=np.float32).transpose((2,0,1))
    img2 = np.array(img2,dtype=np.float32).transpose((2,0,1))
    img = np.concatenate((img1,img2),axis=1)
    label = 0
    res1 = eval(text1)
    res2 = eval(text2)
    if res1 == res2:
        label = 1
    elif res1 < res2:
        label = 0
    else:
        label = 2
    return img, label

class Dataset_num(Dataset):
    def __init__(self, len: int) -> None:
        self.len = len
        # self.data = []
        # self.labels = []
        # for i in range(self.len):
        #     img, label = get_data()
        #     self.data.append(img)
        #     self.labels.append(label)

    def __getitem__(self, index: int):
        img, label = get_data()
        # img, label = self.data[index], self.labels[index]
        return torch.tensor(label, dtype=torch.long), torch.tensor(img, dtype=torch.float32)

    def __len__(self) -> int:
        return self.len


def train():
    train_dataset = Dataset_num(args.data_len)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = Dataset_num(args.data_len)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    model = Net().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # , eps=1e-8)

    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        # =========================train=======================
        for idx, (label, inputs) in enumerate(train_dataloader):
            inputs = inputs.to(args.device)
            label = label.to(args.device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)

        # writer.add_scalar('train/loss', np.average(train_epoch_loss), epoch)
        # writer.add_scalar('train/acc', 100 * acc / nums, epoch)
        writer.add_scalar('reward/classify_reward', 10 * acc / nums - 10, epoch)

        print("epoch = {}, train acc = {:.3f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(train_epoch_loss)))
        # =========================val=========================
        if epoch % 10 != 0:
            continue
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0

            for idx, (label, inputs) in enumerate(val_dataloader):
                inputs = inputs.to(args.device)
                label = label.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)

            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(val_epoch_loss)))

    writer.close()

if __name__ == '__main__':
    train()