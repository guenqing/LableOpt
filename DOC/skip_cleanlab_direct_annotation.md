# 跳过 Cleanlab 直接进入标注（Direct Annotation）

## 目标

在不运行 `RUN ANALYSIS`（Cleanlab 分析）的情况下，允许用户直接进入标注工具，对数据进行人工标注/修正；同时保留原有分析流程不变。

## Dashboard 路径填写规则（必填/选填）

- **必填**
  - `Images Path`
  - `Output Path`
- **选填**
  - `GT Labels Path`
  - `Pred Labels Path`
  - `Human Verified Annotation Path`
  - `Classes File`

## RUN ANALYSIS 的可用条件

只有在同时填写并存在以下三个目录时，才允许执行分析：

- `Images Path`
- `GT Labels Path`
- `Pred Labels Path`

否则点击 `RUN ANALYSIS` 会弹出提示并终止执行。

## 直接 GO TO ANNOTATION TOOL 的样本选择规则

当未运行分析（`analysis_complete=False`）而直接点击 `Go to Annotation Tool` 时：

1. 样本集合始终从 **Images Path** 出发。
2. 如果提供了 `GT Labels Path`，则与 GT label key 做交集（排除 `*_tmp.txt`）。
3. 如果提供了 `Pred Labels Path`，则与 Pred label key 做交集。
4. 最终样本会再排除已处理样本：若 `Output Path` 或 `Human Verified Annotation Path` 中已存在 `{stem}.txt` 或 `{stem}_tmp.txt`，该样本会被跳过。

对应四种组合（都会应用第 4 条跳过规则）：

- a) 仅 `Images Path`：按 Images 全量送入，初始无 GT/Pred 框。
- b) `Images + GT`：取交集送入，初始仅有 GT（editable）。
- c) `Images + Pred`：取交集送入，初始仅有 Pred（reference）。
- d) `Images + GT + Pred`：取三者交集送入，初始同时有 GT（editable）与 Pred（reference）。

## 标注页面框的角色（无需改 UI）

- GT 框：`editable=True`，作为可编辑框（保存时写入 Output 的 `*_tmp.txt`）。
- Pred 框：`editable=False`，作为参考框。

当 `GT/Pred` 路径未配置时，标注页不会再从当前工作目录错误兜底读取标签文件。

## 关键实现点

- 样本选择（直达标注）核心函数：
  - `anno_refiner_app/src/core/file_manager.py::collect_annotation_image_paths`
- Dashboard 分支逻辑：
  - `anno_refiner_app/src/ui/page_dashboard.py`
    - `RUN ANALYSIS` 强制要求 Images+GT+Pred
    - `Go to Annotation Tool`：分析完成走原逻辑；否则走直达标注逻辑
- 新增标注队列类型：
  - `anno_refiner_app/src/models.py::IssueType.DIRECT`
- 标注页读取可选 GT/Pred：
  - `anno_refiner_app/src/ui/page_annotator.py::_load_boxes`

## 测试

- 单元测试目录：`TEST_skip_analysis_direct_annotation/`
  - 覆盖 a/b/c/d 四种路径组合 + Output/HumanVerified 跳过规则
  - 覆盖 `GT/Pred` 路径为空时不从 CWD 兜底读标签
- 返回 Dashboard 交互修复测试：`TEST_back_to_dashboard_fix/`
  - 覆盖确认弹窗按钮不再使用 `asyncio.create_task`（避免丢失 client 上下文导致需要二次点击返回）
- Dashboard 待处理样本数统计：`TEST_pending_sample_count_and_path_parsing/`
  - 覆盖 `Images ∩ GT ∩ Pred - processed(Output/HumanVerified)` 的 pending 统计与边界情况

