#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爬虫脚本 - 用于获取《诡秘之主》小说内容
"""

import json
import os
from typing import Dict, List

# 注意：实际爬虫代码需要根据具体的网站结构编写
# 这里提供一个示例骨架，用于将已有的JSONL文件复制到项目目录

def main():
    # 源文件路径（假设已经爬取到本地）
    source_file = r"c:\Users\Chen\Documents\codes\爬虫\novel_scraper\downloads\诡秘之主_clean.jsonl"
    # 目标路径
    target_file = "../data/raw/诡秘之主.txt"
    
    print("📥 正在将已有数据复制到项目目录...")
    
    # 创建目标目录（如果不存在）
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    
    # 读取源文件并写入目标文件
    with open(source_file, 'r', encoding='utf-8') as src, \
         open(target_file, 'w', encoding='utf-8') as dst:
        # 简单复制内容
        dst.write(src.read())
    
    print(f"✅ 数据已成功复制到 {target_file}")
    print("📝 注意：实际爬虫需要根据网站结构编写，这里仅演示数据迁移")

if __name__ == "__main__":
    main()