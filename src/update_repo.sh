#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

# 检查 git 是否安装
if ! command -v git &> /dev/null; then
    print_message "Error: git is not installed" "$RED"
    exit 1
fi

# 检查是否在 git 仓库中
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    print_message "Error: not a git repository" "$RED"
    exit 1
fi

# 获取当前分支名
current_branch=$(git branch --show-current)
print_message "Current branch: $current_branch" "$YELLOW"

# 检查是否有未提交的更改
if [[ -n $(git status -s) ]]; then
    print_message "Changes detected. Processing..." "$YELLOW"
    
    # 显示更改的文件
    print_message "\nModified files:" "$YELLOW"
    git status -s
    
    # 添加所有更改
    git add .
    
    # 获取当前时间作为提交信息
    commit_date=$(date "+%Y-%m-%d %H:%M:%S")
    
    # 提交更改
    print_message "\nCommitting changes..." "$YELLOW"
    git commit -m "Auto update: $commit_date"
    
    # 推送到远程仓库
    print_message "\nPushing to remote repository..." "$YELLOW"
    if git push origin $current_branch; then
        print_message "\nSuccessfully pushed to remote repository" "$GREEN"
    else
        print_message "\nError: Failed to push to remote repository" "$RED"
        exit 1
    fi
else
    print_message "No changes detected" "$GREEN"
fi

# 显示最新的提交记录
print_message "\nLatest commit:" "$YELLOW"
git log -1 --pretty=format:"%h - %s (%cr) <%an>"
print_message "\n"

exit 0 