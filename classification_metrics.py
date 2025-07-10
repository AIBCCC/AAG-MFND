import torch
from sklearn.metrics import precision_recall_fscore_support

def classification_metrics(predicted, labels):
    """
    计算二分类任务的精度、召回率、F1-score
    
    参数:
    predicted (torch.Tensor): 模型预测的输出,形状为(batch_size,)
    labels (torch.Tensor): 数据的标签,形状为(batch_size,)
    """
    
    # 将预测输出和标签转换为 numpy 数组
    y_pred = predicted.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    
    # 计算分类指标
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # 输出结果
    print("真新闻指标:")
    print(f"Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1-score={f1[0]:.4f}")
    
    print("假新闻指标:")
    print(f"Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1-score={f1[1]:.4f}")