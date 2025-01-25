import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class CurriculumScheduler:
    """课程学习调度器"""
    def __init__(self, num_stages=4, stage_length=1000):
        self.num_stages = num_stages
        self.stage_length = stage_length
        self.current_stage = 0
        self.steps = 0
        
        # 定义课程阶段
        self.curriculum_stages = {
            0: {'name': 'basic', 'constraints': ['budget']},
            1: {'name': 'position', 'constraints': ['budget', 'position']},
            2: {'name': 'team', 'constraints': ['budget', 'position', 'team']},
            3: {'name': 'full', 'constraints': ['budget', 'position', 'team', 'value']}
        }
        
        # 记录每个阶段的性能
        self.stage_performance = {i: deque(maxlen=100) for i in range(num_stages)}
    
    def get_current_constraints(self):
        """获取当前阶段的约束"""
        return self.curriculum_stages[self.current_stage]['constraints']
    
    def update(self, reward):
        """更新课程进度"""
        self.stage_performance[self.current_stage].append(reward)
        self.steps += 1
        
        # 检查是否需要进入下一阶段
        if self._check_stage_completion():
            self.current_stage = min(self.current_stage + 1, self.num_stages - 1)
            return True
        return False
    
    def _check_stage_completion(self):
        """检查当前阶段是否完成"""
        if len(self.stage_performance[self.current_stage]) < 50:
            return False
        
        # 计算最近的平均表现
        avg_reward = np.mean(list(self.stage_performance[self.current_stage]))
        
        # 根据阶段设置不同的晋升阈值
        thresholds = {
            0: 0.6,  # 基础阶段要求较低
            1: 0.7,  # 位置约束阶段
            2: 0.75, # 球队约束阶段
            3: 0.8   # 完整约束阶段
        }
        
        return avg_reward >= thresholds[self.current_stage]

class GradientAggregator:
    """梯度聚合器"""
    def __init__(self, beta=0.9):
        self.beta = beta
        self.grad_buffer = {}
    
    def update_gradients(self, model, loss):
        """更新并聚合梯度"""
        # 计算梯度
        loss.backward()
        
        # 聚合梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.grad_buffer:
                    self.grad_buffer[name] = param.grad.clone()
                else:
                    self.grad_buffer[name] = (
                        self.beta * self.grad_buffer[name] + 
                        (1 - self.beta) * param.grad
                    )
    
    def apply_gradients(self, model):
        """应用聚合后的梯度"""
        for name, param in model.named_parameters():
            if name in self.grad_buffer:
                param.grad = self.grad_buffer[name].clone()

class MetaDistiller(nn.Module):
    """元知识蒸馏器"""
    def __init__(self, inner_loop, outer_loop):
        super().__init__()
        self.inner_loop = inner_loop
        self.outer_loop = outer_loop
        
        # 元知识编码器
        self.meta_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64)
        )
        
        # 知识融合层
        self.knowledge_fusion = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64)
        )
    
    def forward(self, model, batch, meta_batch):
        """
        前向传播，包括内循环和外循环的知识蒸馏
        """
        # 1. 内循环训练
        inner_loss = 0
        constraints = self.inner_loop.get_current_constraints()
        
        for data in batch:
            # 获取当前约束下的输出
            output = model(data, constraints=constraints)
            loss = self._compute_loss(output, data)
            inner_loss += loss
            
            # 更新内循环性能
            self.inner_loop.update(output['reward'])
        
        # 2. 提取元知识
        meta_knowledge = self.meta_encoder(model.get_representation())
        
        # 3. 外循环优化
        outer_loss = 0
        for data in meta_batch:
            # 融合元知识
            current_knowledge = self.meta_encoder(model.get_representation())
            fused_knowledge = self.knowledge_fusion(
                torch.cat([meta_knowledge, current_knowledge], dim=-1)
            )
            
            # 使用融合知识进行预测
            output = model(data, meta_knowledge=fused_knowledge)
            loss = self._compute_loss(output, data)
            outer_loss += loss
            
            # 聚合梯度
            self.outer_loop.update_gradients(model, loss)
        
        # 应用聚合的梯度
        self.outer_loop.apply_gradients(model)
        
        return {
            'inner_loss': inner_loss,
            'outer_loss': outer_loss,
            'meta_knowledge': meta_knowledge
        }
    
    def _compute_loss(self, output, target):
        """计算损失函数"""
        # 主要任务损失
        task_loss = F.mse_loss(output['prediction'], target['value'])
        
        # 约束违反损失
        constraint_loss = sum(output['constraint_violations'].values())
        
        # 知识蒸馏损失
        if 'meta_knowledge' in output:
            kd_loss = F.kl_div(
                F.log_softmax(output['prediction'], dim=-1),
                F.softmax(output['meta_knowledge'], dim=-1)
            )
        else:
            kd_loss = 0
        
        # 总损失
        total_loss = (
            task_loss + 
            0.5 * constraint_loss + 
            0.1 * kd_loss
        )
        
        return total_loss 