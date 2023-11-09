from torch import nn
import torch
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4*32, out_features=64)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 4 * 4 * 32 )
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, hidden_dim, num_heads, num_layers):
        super(ViT, self).__init__()
        self.patch_embedding = nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.randn(num_patches, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, num_heads, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.permute(0, 2, 3, 1).view(x.size(0), -1, x.size(1))
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x
