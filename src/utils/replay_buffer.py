import random
from collections import namedtuple, deque
import numpy as np

# 定义经验元组的结构
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        """初始化缓冲区
        Args:
            capacity (int): 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储一条经验
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """随机采样一批经验
        Args:
            batch_size (int): 批次大小
        Returns:
            Experience: 包含批次数据的经验元组
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # 将经验分解为各个组件
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # 返回命名元组
        return Experience(states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

class PrioritizedReplayBuffer(ReplayBuffer):
    """优先经验回放缓冲区"""
    
    def __init__(self, capacity, alpha=0.6):
        """初始化优先经验回放缓冲区
        Args:
            capacity (int): 缓冲区最大容量
            alpha (float): 优先级的指数参数
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.epsilon = 1e-6  # 防止优先级为0
    
    def push(self, state, action, reward, next_state, done):
        """存储一条经验及其优先级"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)
        super().push(state, action, reward, next_state, done)
    
    def sample(self, batch_size, beta=0.4):
        """基于优先级采样
        Args:
            batch_size (int): 批次大小
            beta (float): 重要性采样的指数参数
        Returns:
            tuple: (经验批次, 重要性权重, 采样的索引)
        """
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(
            len(self.buffer), 
            batch_size, 
            p=probs,
            replace=False
        )
        
        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # 获取经验
        experiences = [self.buffer[idx] for idx in indices]
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return (Experience(states, actions, rewards, next_states, dones), 
                weights, indices)
    
    def update_priorities(self, indices, priorities):
        """更新优先级
        Args:
            indices (list): 需要更新的经验索引
            priorities (list): 新的优先级值
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon 