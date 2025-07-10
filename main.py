from AAR_clip_model import devices
from VLR import VLR
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from Trainer import Trainer
import VLM_MR2_en_dataloader

learning_rate = 2e-5
num_epochs = 40

model = VLR(dim=768).to(devices)        #model = VLR(dim=768).to(devices)

#使用交叉熵作为分类损失函数，适用于二分类或多分类任务
criterion = nn.CrossEntropyLoss()

#使用 AdamW（Adam + 权重衰减）优化器,加了 weight_decay=0.01，用于正则化，防止过拟合
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

#数据加载
batch_size = 16
#输出格式(x1, x2, x3, x4, labels)
train_loader = VLM_MR2_en_dataloader.load_train_MR2(batch_size)
test_loader = VLM_MR2_en_dataloader.load_test_MR2(batch_size)

trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, device=devices)
trainer.fit(num_epochs)

# 保存模型
#torch.save(model.state_dict(), "vlr_model_best.pth")

# # 加载模型
# model = VLR(dim=768).to(devices)
# model.load_state_dict(torch.load("vlr_model_best.pth", map_location=devices))
# model.eval()  # 加载后切换为评估模式
