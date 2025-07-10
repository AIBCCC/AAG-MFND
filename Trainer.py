import torch.optim as optim
from CLIP_pipeline import CLIP_pipeline
import torch
from classification_metrics import classification_metrics
import csv
import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        self.model = model                  #VLR模型
        self.train_loader = train_loader    #训练数据加载器
        self.test_loader = test_loader      #测试数据加载器
        self.criterion = criterion          #损失函数
        self.optimizer = optimizer          #优化器
        self.device = device                #运行设备
        self.scaler = GradScaler()          #创建梯度缩放器用于AMP训练
     
    def train_epoch(self):
        self.model.train()                  #设置为训练模式
        running_loss = 0.0                  #初始化损失累计变量
        
        #从训练集取出一批数据
        for x1, x2, x3, x4, labels in self.train_loader:

            # 使用AMP进行CLIP处理
            with autocast():
                x1, x2, x3, x4 = CLIP_pipeline(x1, x2, x3, x4)              #用 CLIP_pipeline 抽取四种特征（文本、图像、正/反推理文本）
                x1, x2, x3, x4, labels = x1.to(self.device), x2.to(self.device), x3.to(self.device), x4.to(self.device),labels.to(self.device)    #送入设备（GPU）
                
                self.optimizer.zero_grad()                                  #清空梯度

            # 使用autocast上下文进行前向传播
            with autocast():
                outputs_1, outputs = self.model(x1, x2, x3, x4)             #前向传播：outputs_1：第一次判断（融合图文），outputs：第二次判断（结合对抗推理）

                loss = self.criterion(outputs, labels) + self.criterion(outputs_1, labels)      #损失函数为两个判断输出的总和（等权监督）

            # loss.backward()             #反向传播
            # self.optimizer.step()       #参数更新
            # 使用scaler进行梯度缩放、反向传播和优化
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
            running_loss += loss.item() * x1.size(0)                        #累计损失
        epoch_loss = running_loss / len(self.train_loader.dataset)      #求平均
        return epoch_loss

    def test(self):
        self.model.eval()       #模型设为评估模式
        running_loss = 0.0      #初始化准确率统计用变量
        correct = 0
        total = 0

        # 累积预测输出和标签
        all_predicted = []
        all_labels = []

        with torch.no_grad():
            for x1, x2, x3, x4, labels in self.test_loader:
                with autocast():
                    x1, x2, x3, x4 = CLIP_pipeline(x1, x2, x3, x4)
                    x1, x2, x3, x4, labels = x1.to(self.device), x2.to(self.device), x3.to(self.device), x4.to(self.device),labels.to(self.device)  # to device
                
                with autocast():
                    outputs_1, outputs = self.model(x1, x2, x3, x4)

                    loss = self.criterion(outputs, labels)+self.criterion(outputs_1, labels)

                running_loss += loss.item() * x1.size(0)            #loss.item()：把当前 batch 的平均损失（标量）取出来；x1.size(0)：当前 batch 的样本数（即 batch_size）
                _, predicted = torch.max(outputs, 1)                #outputs的shape 为 [batch_size, num_classes],行为batch数，列为分类个数，max取在分类个数维度上预测分数最大的类别
                total += labels.size(0)                             #labels.size(0) 是当前 batch 的样本数，total 用来累计整个 epoch 处理过的样本数
                correct += (predicted == labels).sum().item()       #统计预测正确的个数，.item()：转换为 Python 标量

                # 将预测输出和标签添加到列表中
                all_predicted.extend(predicted.cpu())
                all_labels.extend(labels.cpu())

        epoch_loss = running_loss / len(self.test_loader.dataset)
        accuracy = correct / total
        
        # 调用 classification_metrics 函数
        classification_metrics(torch.tensor(all_predicted), torch.tensor(all_labels))

        # # 保存预测结果
        # import pandas as pd
        # df = pd.DataFrame({
        #     "true_label": torch.tensor(all_labels).numpy(),
        #     "predicted_label": torch.tensor(all_predicted).numpy()
        # })
        # df.to_csv("test_predictions.csv", index=False)

        return epoch_loss, accuracy

    def fit(self, epochs, patience=5):
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):                 #每轮训练一个epoch
            train_loss = self.train_epoch()         #训练集损失
            val_loss, val_accuracy = self.test()    #验证集损失与精确率
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            #early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "AAR/best_model.pt")  # 保存最优模型
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        # 保存为 CSV
        # log=[]
        # with open("training_log.csv", "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])
        #     writer.writerows(log)