"""
Vision Transformer (ViT) 训练和评估模块
包含完整的训练循环、数据处理和模型评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from vit_model import create_vit_model, count_parameters

class SimpleImageDataset(Dataset):
    """
    简单的图像数据集
    用于演示和教学目的
    """
    
    def __init__(self, num_samples=1000, img_size=224, num_classes=10, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform
        
        # 生成随机数据
        np.random.seed(42)  # 确保可重复
        self.data = []
        self.labels = []
        
        for i in range(num_samples):
            # 创建带有模式的合成图像
            img = self._create_synthetic_image(i % num_classes)
            self.data.append(img)
            self.labels.append(i % num_classes)
    
    def _create_synthetic_image(self, class_id):
        """创建合成图像，每个类别有不同的模式"""
        img = np.random.rand(self.img_size, self.img_size, 3) * 0.3
        
        # 为不同类别添加不同的模式
        if class_id == 0:  # 红色方块
            img[50:150, 50:150, 0] = 1.0
        elif class_id == 1:  # 绿色圆形
            center = self.img_size // 2
            y, x = np.ogrid[:self.img_size, :self.img_size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= 50 ** 2
            img[mask, 1] = 1.0
        elif class_id == 2:  # 蓝色条纹
            img[::10, :, 2] = 1.0
        elif class_id == 3:  # 黄色对角线
            for i in range(self.img_size):
                if i < self.img_size:
                    img[i, i, :2] = 1.0
        elif class_id == 4:  # 紫色网格
            img[::20, :, [0, 2]] = 0.8
            img[:, ::20, [0, 2]] = 0.8
        elif class_id == 5:  # 橙色三角形
            for i in range(self.img_size):
                for j in range(i):
                    if i + j < self.img_size:
                        img[i, j, [0, 1]] = [1.0, 0.5]
        elif class_id == 6:  # 青色星形
            center = self.img_size // 2
            for angle in [0, 72, 144, 216, 288]:
                rad = np.radians(angle)
                for r in range(0, 60, 2):
                    x = int(center + r * np.cos(rad))
                    y = int(center + r * np.sin(rad))
                    if 0 <= x < self.img_size and 0 <= y < self.img_size:
                        img[y, x, [1, 2]] = 1.0
        elif class_id == 7:  # 棕色棋盘
            for i in range(0, self.img_size, 20):
                for j in range(0, self.img_size, 20):
                    if (i // 20 + j // 20) % 2 == 0:
                        img[i:i+20, j:j+20, [0, 1]] = [0.6, 0.3]
        elif class_id == 8:  # 粉色波浪
            for i in range(self.img_size):
                wave = int(self.img_size // 2 + 30 * np.sin(2 * np.pi * i / 50))
                if 0 <= wave < self.img_size:
                    img[wave-2:wave+2, i, [0, 2]] = [1.0, 0.7]
        else:  # 灰色同心圆
            center = self.img_size // 2
            y, x = np.ogrid[:self.img_size, :self.img_size]
            for r in range(20, 100, 20):
                mask = ((x - center) ** 2 + (y - center) ** 2 >= (r-5) ** 2) & \
                       ((x - center) ** 2 + (y - center) ** 2 <= r ** 2)
                img[mask] = 0.5
        
        return (img * 255).astype(np.uint8)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # 转换为PIL图像
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class ViTTrainer:
    """
    ViT 训练器
    提供完整的训练、验证和测试功能
    """
    
    def __init__(self, model, device='cpu', save_dir='vit_checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def prepare_data(self, batch_size=32, num_samples=1000, val_split=0.2):
        """准备训练和验证数据"""
        
        # 数据变换
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        full_dataset = SimpleImageDataset(
            num_samples=num_samples, 
            transform=train_transform
        )
        
        # 分割训练和验证集
        val_size = int(val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 为验证集设置不同的变换
        val_dataset.dataset.transform = val_transform
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return self.train_loader, self.val_loader
    
    def train_epoch(self, optimizer, criterion, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f'训练 Epoch {epoch}: [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)] '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, criterion):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs=10, learning_rate=1e-4, weight_decay=1e-4):
        """完整训练流程"""
        print("=" * 60)
        print("开始 ViT 训练")
        print("=" * 60)
        
        # 打印模型信息
        total_params, trainable_params = count_parameters(self.model)
        print(f"模型参数数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 设置优化器和损失函数
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        # 训练循环
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
            
            # 验证
            val_loss, val_acc = self.validate(criterion)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            
            # 打印结果
            epoch_time = time.time() - epoch_start
            print(f'\nEpoch {epoch}/{epochs}:')
            print(f'  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
            print(f'  验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
            print(f'  学习率: {optimizer.param_groups[0]["lr"]:.2e}')
            print(f'  耗时: {epoch_time:.2f}s')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, 'best_model.pth')
                print(f'  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)')
            
            print('-' * 60)
        
        total_time = time.time() - start_time
        print(f'\n训练完成! 总耗时: {total_time:.2f}s')
        print(f'最佳验证准确率: {best_val_acc:.2f}%')
        
        return self.train_history
    
    def save_checkpoint(self, epoch, accuracy, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'train_history': self.train_history
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        return checkpoint['epoch'], checkpoint['accuracy']
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_history['loss'], label='训练损失', color='blue')
        ax1.plot(self.train_history['val_loss'], label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_history['accuracy'], label='训练准确率', color='blue')
        ax2.plot(self.train_history['val_accuracy'], label='验证准确率', color='red')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300)
        plt.show()
    
    def evaluate_model(self, test_loader=None):
        """评估模型性能"""
        if test_loader is None:
            test_loader = self.val_loader
        
        self.model.eval()
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 按类别统计
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_correct[label] = class_correct.get(label, 0) + (predicted[i] == target[i]).item()
                    class_total[label] = class_total.get(label, 0) + 1
        
        overall_accuracy = 100 * correct / total
        print(f'\n整体准确率: {overall_accuracy:.2f}%')
        
        print('\n各类别准确率:')
        for class_id in sorted(class_correct.keys()):
            if class_total[class_id] > 0:
                acc = 100 * class_correct[class_id] / class_total[class_id]
                print(f'  类别 {class_id}: {acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})')
        
        return overall_accuracy


def demo_training():
    """演示完整的训练流程"""
    print("=" * 60)
    print("ViT 训练演示")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_vit_model(
        model_name='vit_tiny',  # 使用小模型便于演示
        img_size=224,
        num_classes=10
    )
    
    # 创建训练器
    trainer = ViTTrainer(model, device)
    
    # 准备数据
    trainer.prepare_data(batch_size=16, num_samples=200)  # 少量数据便于演示
    
    # 开始训练
    history = trainer.train(epochs=5, learning_rate=1e-3)
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 评估模型
    trainer.evaluate_model()
    
    return trainer, history


def compare_model_sizes():
    """比较不同大小模型的性能"""
    print("=" * 60)
    print("比较不同 ViT 模型大小")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_names = ['vit_tiny', 'vit_small']
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'-' * 40}")
        print(f"测试 {model_name}")
        print(f"{'-' * 40}")
        
        # 创建模型
        model = create_vit_model(model_name=model_name, num_classes=10)
        total_params, _ = count_parameters(model)
        
        # 创建训练器
        trainer = ViTTrainer(model, device)
        trainer.prepare_data(batch_size=16, num_samples=100)
        
        # 快速训练
        start_time = time.time()
        history = trainer.train(epochs=3, learning_rate=1e-3)
        train_time = time.time() - start_time
        
        # 评估
        final_acc = trainer.evaluate_model()
        
        results[model_name] = {
            'params': total_params,
            'train_time': train_time,
            'accuracy': final_acc,
            'final_train_acc': history['accuracy'][-1],
            'final_val_acc': history['val_accuracy'][-1]
        }
    
    # 输出比较结果
    print("\n" + "=" * 60)
    print("模型比较结果")
    print("=" * 60)
    print(f"{'模型':<12} {'参数数量':<12} {'训练时间':<10} {'最终准确率':<12}")
    print("-" * 60)
    
    for model_name, result in results.items():
        print(f"{model_name:<12} {result['params']:<12,} {result['train_time']:<10.1f}s {result['accuracy']:<12.2f}%")
    
    return results


if __name__ == "__main__":
    # 演示完整训练
    trainer, history = demo_training()
    
    # 比较不同模型
    # comparison_results = compare_model_sizes()
    
    print("\n" + "=" * 60)
    print("ViT 训练演示完成！")
    print("查看生成的图表和检查点文件")
    print("=" * 60)
