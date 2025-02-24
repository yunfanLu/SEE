#!/bin/bash

# 起始数字
start_num=168

# 结束数字
end_num=172

# 循环创建文件夹
for (( num = start_num; num <= end_num; num++ ))
do
    # 使用 printf 格式化数字，确保它是两位数，不足两位数时前面补0
    printf -v formatted_num "%03d" "$num"

    # 创建文件夹，文件夹名称为 "编号-indoor"
    mkdir "$formatted_num-outdoor"
done

echo "文件夹创建完成。"