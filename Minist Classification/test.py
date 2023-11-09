import torch
import torchvision
from torchvision.transforms import transforms
from model import MyNet
from model import ViT
# 绘制分类错误的图像
import numpy as np
from matplotlib import pyplot as plt

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 加载模型
net1 = ViT(28,4,10,32,8,4)
net2=MyNet()
net1.load_state_dict(torch.load('mnist_vit.pth'))

batch_size=64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
# 测试模型
correct = 0
total = 0
incorrect_images = []
incorrect_labels = []
correct_labels=[]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                incorrect_images.append(images[i])
                incorrect_labels.append(predicted[i])
                correct_labels.append(labels[i])

print(f'Accuracy on the test set: {100 * correct / total}%')
for i in range(10):  # 绘制前10个分类错误的图像
    print(f'Predicted: {incorrect_labels[i]}')
    print(f'true: {correct_labels[i]}')
    imshow(incorrect_images[i])

