#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗脚本 - 用于清洗爬取的小说内容
"""

import json
import os
import re

def strict_clean(text: str, title: str) -> str:
    """
    严格清洗文本，移除分隔符和重复标题
    
    Args:
        text: 原始文本
        title: 章节标题
        
    Returns:
        清洗后的纯净文本
    """
    # 1. 移除底部的分隔符
    text = text.replace("------------------------------", "")
    
    # 2. 移除开头重复的标题 (如果存在)
    lines = text.split('\n')
    if len(lines) > 0 and (title in lines[0] or lines[0].strip() in title):
        text = '\n'.join(lines[1:])
        
    return text.strip()

def main():
    input_file = "../data/raw/诡秘之主.txt"
    output_file = "../data/processed/诡秘之主_final.jsonl"

    
    print("🧹 开始执行数据清洗...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 执行清洗
                data['text'] = strict_clean(data['text'], data['title'])
                
                # 写入新文件
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
            except json.JSONDecodeError as e:
                print(f"❌ 解析错误: {e}, 跳过该行")
    
    print(f"✅ 最终清洗完成！共处理 {processed_count} 章。")
    print(f"📁 清洗后的数据已保存到: {output_file}")

if __name__ == "__main__":
    main()