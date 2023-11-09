import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import MyNet
from model import ViT

# 数据加载和预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


# 构建神经网络模型

net1=ViT(28,4,10,32,8,4)
net2 =   MyNet()


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net1.parameters(), lr=0.01, momentum=0.9)

# 学习率调度
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)



# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    with open("vit_training_records.txt", "a") as f:
        f.write(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}\n')
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

print("Finished Training")
# 保存和加载模型
torch.save(net1.state_dict(), 'mnist_vit.pth')
