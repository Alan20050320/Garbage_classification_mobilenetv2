import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. 增强数据预处理 - 关键：更强的正则化
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # 随机裁剪，模拟不同尺度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动，提升泛化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 2. 加载数据集（保持不变）
train_data = datasets.ImageFolder('garbage_classification', transform=train_transform)
val_data = datasets.ImageFolder('garbage_classification', transform=val_transform)
class_names = train_data.classes
print("类别:", class_names)

n_total = len(train_data)
n_train = int(n_total * 0.8)
n_val = n_total - n_train
indices = torch.randperm(n_total).tolist()
train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_set = Subset(train_data, train_indices)
val_set = Subset(val_data, val_indices)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)  # 增大batch_size
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
print(f"训练集:{n_train}张,验证集:{n_val}张")

# 3. 改进的模型加载策略 - 核心改动
model = models.mobilenet_v2(weights='IMAGENET1K_V1')

# 策略：解冻最后3个倒残差块（共7个），让模型适应新任务
# MobileNetV2的features包含多个倒残差块，我们解冻最后一部分
layers_to_unfreeze = 3
total_blocks = len([m for m in model.features if isinstance(m, nn.Conv2d)])
print(f"模型总卷积块数: {total_blocks}, 解冻最后 {layers_to_unfreeze} 个块")

# 冻结大部分层
for param in model.features.parameters():
    param.requires_grad = False

# 解冻最后几层
for i, child in enumerate(reversed(list(model.features.children()))):
    if i < layers_to_unfreeze:
        for param in child.parameters():
            param.requires_grad = True

# 替换分类头
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
print("模型加载完成，开始训练...")

# 4. 改进的训练配置
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"正在使用的设备: {device}")

model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑，防止过拟合

# 使用AdamW优化器（带权重衰减的Adam），效果更好
optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': 0.0001},  # 解冻层用较小学习率
    {'params': model.classifier.parameters(), 'lr': 0.001}  # 分类头用正常学习率
], weight_decay=1e-4)  # 权重衰减，正则化

# 余弦退火学习率调度，自动调整学习率
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

# 5. 训练循环（保持20轮，但加入早停）
EPOCHS = 20
best_acc = 0.0
best_epoch = 0
patience = 5  # 早停耐心值
no_improve = 0

print("\n开始训练（改进版）...")
print("-" * 80)

for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / n_train
    avg_loss = total_loss / len(train_loader)

    # 更新学习率
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # 验证阶段
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / n_val

    # 输出详细信息
    status = ""
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        no_improve = 0
        torch.save(model.state_dict(), 'best_model_v2.pth')
        status = "✓ 保存最佳模型"
    else:
        no_improve += 1
        if no_improve >= patience:
            status = f"⚠ 早停计数: {no_improve}/{patience}"
        else:
            status = ""

    print(f"Epoch[{epoch + 1:2d}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} "
          f"Train: {train_acc:.3f} "
          f"Val: {val_acc:.3f} "
          f"LR: {current_lr:.5f}  {status}")

    # 早停检查
    if no_improve >= patience:
        print(f"\n⏹ 早停触发: 连续{patience}轮验证准确率未提升")
        break

print("-" * 80)
print(f"\n训练完成!")
print(f"最优验证准确率: {best_acc:.4f} (第{best_epoch}轮)")
print(f"模型已保存至: best_model_v2.pth")