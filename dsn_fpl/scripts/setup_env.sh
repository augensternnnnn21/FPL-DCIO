#!/bin/bash

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装基本依赖
pip install -r requirements.txt

# 如果是开发环境，安装开发依赖
if [ "$1" = "dev" ]; then
    pip install -r requirements-dev.txt
fi

# 安装pre-commit hooks
if [ "$1" = "dev" ]; then
    pre-commit install
fi

# 创建.env文件
if [ ! -f .env ]; then
    cp .env.example .env
fi 