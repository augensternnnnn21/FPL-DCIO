import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        """初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            ttl: 缓存过期时间（秒）
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def _get_cache_path(self, key: str) -> str:
        """获取缓存文件路径"""
        # 使用key的md5值作为文件名，避免文件名过长或包含特殊字符
        import hashlib
        filename = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{filename}.json")
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据，如果不存在或已过期则返回None
        """
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                
            # 检查是否过期
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                os.remove(cache_path)  # 删除过期缓存
                return None
                
            return cache_data['data']
        except Exception as e:
            print(f"读取缓存出错: {str(e)}")
            return None
            
    def set(self, key: str, data: Any):
        """设置缓存数据
        
        Args:
            key: 缓存键
            data: 要缓存的数据
        """
        cache_path = self._get_cache_path(key)
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"写入缓存出错: {str(e)}")
            
    def clear(self):
        """清除所有缓存"""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"清除缓存出错: {str(e)}")
            
    def clear_expired(self):
        """清除过期缓存"""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        cache_data = json.load(f)
                    cached_time = datetime.fromisoformat(cache_data['timestamp'])
                    if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                        os.remove(file_path)
        except Exception as e:
            print(f"清除过期缓存出错: {str(e)}") 