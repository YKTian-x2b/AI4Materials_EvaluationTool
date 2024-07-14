# 显示10个最大的文件id列表
git verify-pack -v .git/objects/pack/pack-*.idx | sort -k 3 -g | tail -10

# 根据文件id找出文件所在路径
git rev-list --objects --all | grep {文件id}

# 删除文件
git log --pretty=oneline --branches -- {文件路径}

# 删除文件的历史记录
git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch {文件路径}' --prune-empty --tag-name-filter cat -- --all

# 清除缓存（真正删除）
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now
git push origin main

# 让远程仓库变小
git remote prune origin
