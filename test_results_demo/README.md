# 测试结果展示 - Run ID: 20260108_202510_0fdc6d

## 运行信息
- **运行ID**: 20260108_202510_0fdc6d
- **运行时间**: 2026-01-08 20:25:10 - 22:09:05
- **翻译章节**: 第1-10章
- **总耗时**: 6235.07秒（约104分钟）

## 文件说明

### 1. glossary_v1.json
完整术语表，包含33个术语条目，每个条目包含：
- 术语原文和类型（NE:person, NE:location, domain_term等）
- 最终译名（final）
- 候选翻译列表（candidates，带评分）
- 证据片段（evidence_span）
- 多义词义项（senses，当前为空）

### 2. run.log
运行日志，记录工作流各节点的执行情况

### 3. book_merged_zh.txt
最终合并的翻译结果（前10章）

### 4. sample_chunk.json
示例chunk翻译结果，展示：
- 原文（source）
- 初稿（draft）
- 回译（back_translation）
- 审校意见（critique）
- 最终翻译（final_translation）
- 术语违规检查（violations）

## 关键验证点

1. **术语一致性**: 所有chunk的violations字段为空或仅包含"needs_refine"，说明术语使用正确
2. **术语分类**: glossary中包含7种类型的术语（NE:person, NE:location, NE:org, NE:deity, NE:language, NE:identity, domain_term）
3. **结构化存储**: 每个术语条目包含完整的结构化字段

