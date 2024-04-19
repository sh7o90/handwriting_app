import torch
import torchvision
from torchvision import transforms, datasets
import os

# データセットの変換を定義
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 白黒画像を3チャンネルのRGB画像に変換
    transforms.Resize((224, 224)),  # EfficientNetV2の入力サイズにリサイズ
    transforms.RandomHorizontalFlip(p=1),  # 必ず水平方向に反転
    transforms.RandomRotation(degrees=(90, 90)),  # 必ず90度時計回りに回転
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # MNISTの平均と標準偏差を使用
])

data_dir = './'

# EMNISTデータセットをロードしてDataLoaderを定義
batch_size = 32

train_dataset = datasets.EMNIST(root=data_dir, split='balanced', train=True, download=False, transform=transform)
train_loader_transposed = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = datasets.EMNIST(root=data_dir, split='balanced', train=False, download=False, transform=transform)
val_loader_transposed = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

test_dataset = datasets.EMNIST(root=data_dir, split='balanced', train=False, download=False, transform=transform)
test_loader_transposed = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


