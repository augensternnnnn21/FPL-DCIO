import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        """初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            ttl: 缓存过期时间（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用MD5生成文件名，避免特殊字符
        filename = hashlib.md5(key.encode()).hexdigest() + '.json'
        return self.cache_dir / filename
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        cache_path = self._get_cache_path(key)
        
        try:
            if not cache_path.exists():
                return None
                
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # 检查是否过期
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                self._remove_cache(cache_path)
                return None
                
            return cache_data['data']
            
        except Exception as e:
            print(f"读取缓存出错: {str(e)}")
            return None
            
    def set(self, key: str, value: Any):
        """设置缓存数据"""
        cache_path = self._get_cache_path(key)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': value
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"写入缓存出错: {str(e)}")
            
    def clear(self):
        """清除所有缓存"""
        try:
            for cache_file in self.cache_dir.glob('*.json'):
                self._remove_cache(cache_file)
        except Exception as e:
            print(f"清除缓存出错: {str(e)}")
            
    def clear_expired(self):
        """清除过期缓存"""
        try:
            for cache_file in self.cache_dir.glob('*.json'):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        
                    cached_time = datetime.fromisoformat(cache_data['timestamp'])
                    if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                        self._remove_cache(cache_file)
                        
                except Exception:
                    # 如果读取失败，删除可能损坏的缓存文件
                    self._remove_cache(cache_file)
                    
        except Exception as e:
            print(f"清除过期缓存出错: {str(e)}")
            
    def _remove_cache(self, cache_path: Path):
        """删除缓存文件"""
        try:
            cache_path.unlink()
        except Exception as e:
            print(f"删除缓存文件出错: {str(e)}") 