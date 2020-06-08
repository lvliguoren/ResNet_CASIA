from imageset import customData
import torch
import torch.utils.data as Data
import torchvision
import codecs
import os
from torch.utils.tensorboard import SummaryWriter

index_path = 'CASIA_index'
local_dir = 'E:/TEST/'

with codecs.open(os.path.join(index_path, 'train_img.txt'), 'r', 'utf-8') as code_file:
    train_images =[i.replace('../../', local_dir) for i in code_file.read().split(',')]

with codecs.open(os.path.join(index_path, 'train_lbl.txt'), 'r', 'utf-8') as code_file:
    train_labels = [int(i) for i in code_file.read().split(',')]

with codecs.open(os.path.join(index_path, 'test_img.txt'), 'r', 'utf-8') as code_file:
    test_images = [i.replace('../../', local_dir) for i in code_file.read().split(',')]

with codecs.open(os.path.join(index_path, 'test_lbl.txt'), 'r', 'utf-8') as code_file:
    test_labels = [int(i) for i in code_file.read().split(',')]

train_set = customData(images_path=train_images, labels= train_labels)
train_loader = Data.DataLoader(
    dataset= train_set,
    shuffle=True,
    batch_size=128,
)

test_set = customData(images_path=test_images, labels=test_labels)
test_loader = Data.DataLoader(
    dataset = test_set,
    shuffle = True,
    batch_size = 500,
)

model = torchvision.models.resnet18(pretrained=False, num_classes=3755)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_func = torch.nn.CrossEntropyLoss()

def train(train_x, train_y):
    model.train()

    outputs = model(train_x)
    loss = loss_func(outputs, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 转为2维才能和同样是2维的top5对比
    y_resize = train_y.view(-1, 1)
    _, top5 = torch.topk(outputs, 5, 1)
    accuracy = torch.eq(top5, y_resize).sum().float().mean()

    return loss.item(), accuracy.item()

@torch.no_grad()
def test():
    model.eval()

    accuracy = 0.0
    for test_x, test_y in test_loader:
        test_outputs = model(test_x)
        y_resize = test_y.view(-1, 1)
        _, top5 = torch.topk(test_outputs, 5, 1)
        accuracy = torch.eq(top5, y_resize).sum().float().mean()
        # 只取第一个batch作为测试，避免耗时过长
        break
    return accuracy.item()

with SummaryWriter() as writer:
    for step, (batch_x, batch_y) in enumerate(train_loader):
        train_loss, train_accuracy = train(batch_x, batch_y)

        if step % 10 == 0:
            test_accuracy = test()
            print('step:{},train_loss:{},train_accuracy:{},test_accuracy:{}'.format(step,train_loss,train_accuracy,test_accuracy))

            writer.add_scalar('Loss/train', train_loss, step)
            writer.add_scalars('Accuracy', {"train": train_accuracy, "test": test_accuracy}, step)

        if step % 500 == 0:
            torch.save(model, "model.pth")