#!/bin/bash
# 清理所有缓存和临时文件
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +
rm -rf experiments/ 