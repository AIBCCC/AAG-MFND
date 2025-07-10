import CLIP_pipeline
import torch

# 支持DeepSpeed的训练器
class DeepSpeedTrainer:
    def __init__(self, model_engine, train_loader, test_loader, criterion, local_rank=-1, is_distributed=False):
        self.model_engine = model_engine        #使用 DeepSpeed 初始化后的模型，封装了前向、反向和优化过程
        self.train_loader = train_loader        #PyTorch 的训练数据加载器（支持分布式 sampler）
        self.test_loader = test_loader          #PyTorch 的测试数据加载器
        self.criterion = criterion              #损失函数（如 nn.CrossEntropyLoss()）
        self.local_rank = local_rank            #当前进程的 GPU 编号，DeepSpeed 使用
        self.is_distributed = is_distributed    #是否为多卡分布式训练
        
        # 记录最佳性能
        self.best_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        self.model_engine.train()
        running_loss = 0.0
        
        # 在每个epoch开始时设置DistributedSampler的epoch,保证多卡分布式训练中 DistributedSampler 每轮随机性一致（确保打乱数据）。
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        for x1, x2, x3, x4, labels in self.train_loader:
            # 使用CLIP处理数据
            x1, x2, x3, x4 = CLIP_pipeline(x1, x2, x3, x4)
            
            # 将数据移到模型所在设备
            x1 = x1.to(self.model_engine.device)
            x2 = x2.to(self.model_engine.device)
            x3 = x3.to(self.model_engine.device)
            x4 = x4.to(self.model_engine.device)
            labels = labels.to(self.model_engine.device)
            
            # 前向传播
            outputs_1, outputs = self.model_engine(x1, x2, x3, x4)
            loss = self.criterion(outputs, labels) + self.criterion(outputs_1, labels)
            
            # DeepSpeed管理反向传播和优化器步骤
            self.model_engine.backward(loss)
            self.model_engine.step()
            
            # 记录损失
            running_loss += loss.item() * x1.size(0)
        
        # 计算整个epoch的平均损失
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    def test(self):
        self.model_engine.eval()
        running_loss = 0.0      #初始化准确率统计用变量
        correct = 0
        total = 0
        
        #防止梯度追踪，加快推理过程
        with torch.no_grad():
            for x1, x2, x3, x4, labels in self.test_loader:
                # 处理数据
                x1, x2, x3, x4 = CLIP_pipeline(x1, x2, x3, x4)
                
                # 将数据移到设备上
                x1 = x1.to(self.model_engine.device)
                x2 = x2.to(self.model_engine.device)
                x3 = x3.to(self.model_engine.device)
                x4 = x4.to(self.model_engine.device)
                labels = labels.to(self.model_engine.device)
                
                # 前向传播
                outputs_1, outputs = self.model_engine(x1, x2, x3, x4)
                _, predicted = torch.max(outputs.data, 1)                                   #获取预测分类（最大概率索引）
                loss = self.criterion(outputs, labels)+self.criterion(outputs_1, labels)
                
                # 计算准确率
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item() * x1.size(0) 
        
        epoch_loss = running_loss / len(self.test_loader.dataset)
        accuracy = correct / total
        return epoch_loss,accuracy
    
    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy = self.test()
            # 在主进程上进行测试和打印
            if self.local_rank == 0 or self.local_rank == -1:
                accuracy = self.test()
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
                
                # 检查是否是最佳模型
                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    self.best_epoch = epoch
                    
                    # 保存最佳模型检查点
                    self.save_checkpoint("vlr_model_best_DS")
            
            # 同步所有进程
            if self.is_distributed:
                torch.distributed.barrier()
    
    def save_checkpoint(self, tag):
        """保存检查点"""
        if self.local_rank == 0 or self.local_rank == -1:  # 只在主进程保存
            client_state = {'best_acc': self.best_acc, 'best_epoch': self.best_epoch}
            self.model_engine.save_checkpoint(save_dir="checkpoints", tag=tag, client_state=client_state)
            print(f"Model saved at epoch {self.best_epoch+1} with accuracy {self.best_acc:.4f}")
