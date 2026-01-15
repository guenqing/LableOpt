# Refiner 系统开发状态追踪

> 最后更新: 2026-01-15

## 项目概述

Refiner 是一个面向 YOLO 目标检测标签修正工作的 Web UI 工具，基于 Cleanlab 自动定位可能有误的标注样本，支持人工审核和修正。

### 解决的三类标签问题

| 问题类型 | 说明 | 检测原理 |
|---------|------|---------|
| **Overlooked（漏标）** | GT 中遗漏了应标注的目标 | 模型高置信度预测了框，但 GT 无对应标注 |
| **Swapped（类别错误）** | GT 中目标的类别标注错误 | GT 框类别与模型预测类别不一致 |
| **Bad Located（框不准）** | GT 中边界框位置/尺寸偏差大 | GT 框与模型预测框的 IoU 较低 |

---

## 开发阶段状态

| 阶段 | 内容 | 状态 | 完成日期 |
|-----|------|------|---------|
| 阶段一 | 环境搭建与基础验证 | 已完成 | 2026-01-15 |
| 阶段二 | 核心工具模块 (yolo_utils, file_manager) | 已完成 | 2026-01-15 |
| 阶段三 | Cleanlab 分析器 | 已完成 | 2026-01-15 |
| 阶段四 | 页面A - Dashboard | 已完成 | 2026-01-15 |
| 阶段五 | 标注组件开发 | 未开始 | - |
| 阶段六 | 页面B - Annotator | 未开始 | - |
| 阶段七 | 集成测试与优化 | 未开始 | - |

---

## 已完成功能详情

### 阶段一：环境搭建与基础验证

- [x] 创建项目目录结构 `anno_refiner_app/`
- [x] 验证 NiceGUI 导入正常
- [x] 验证 Cleanlab object_detection 模块导入正常
- [x] 创建数据模型 `models.py`

### 阶段二：核心工具模块

#### `src/core/yolo_utils.py`

- [x] `yolo_to_pixel()` / `pixel_to_yolo()`: YOLO 归一化坐标与像素坐标互转
- [x] `read_yolo_label()`: 读取 YOLO 标签文件（支持 GT 和 Pred 格式）
- [x] `write_yolo_label()`: 写入 YOLO 标签文件
- [x] `collect_image_paths()`: 递归收集嵌套目录结构中的图片路径
- [x] `prepare_cleanlab_labels()`: 将 GT 标签转换为 Cleanlab 格式
- [x] `prepare_cleanlab_predictions()`: 将预测标签转换为 Cleanlab 格式
- [x] `count_classes()`: 统计类别数量

#### `src/core/file_manager.py`

- [x] `backup_folder()`: 文件夹备份（支持时间戳后缀）
- [x] `save_tmp_annotation()`: 保存临时标注文件
- [x] `confirm_changes()`: 确认或放弃标注更改
- [x] `get_tmp_files()`: 获取所有临时文件列表
- [x] `validate_paths()`: 验证输入路径并返回统计信息

### 阶段三：Cleanlab 分析器

#### `src/core/analyzer.py`

- [x] `CleanlabAnalyzer` 类实现
- [x] `prepare_data()`: 加载数据并转换为 Cleanlab 格式
- [x] `analyze(top_k)`: 执行分析并返回 TopK 问题样本
- [x] `get_label_quality_scores()`: 获取所有图片的整体质量分数
- [x] 支持嵌套目录结构 `{category}/{video}/frame_xxxxx.jpg`
- [x] 进度回调支持
- [x] NaN 值过滤（完美匹配的框不报告为问题）

### 阶段四：页面A - Dashboard

#### `src/state.py`

- [x] `AppState` 全局状态管理类
- [x] `AnalysisResults` 分析结果存储
- [x] `get_selected_issues()`: 合并选中类型的问题（去重）

#### `src/ui/page_dashboard.py`

- [x] 路径输入与实时验证（显示文件数量）
- [x] Cleanlab 分析触发与进度条显示
- [x] 三列问题列表展示（固定高度，可滚动）
- [x] TopK 配置与刷新按钮
- [x] 问题类型勾选（Overlooked/Swapped/Bad Located）
- [x] **点击即时可视化**：点击列表项生成临时可视化图片
- [x] 可视化面板：显示 GT 框（绿色）、Pred 框（蓝色虚线）、问题框（红色高亮）
- [x] 备份选项（默认关闭）
- [x] "Go to Annotation Tool" 跳转按钮

#### `main.py`

- [x] NiceGUI 应用入口
- [x] 路由配置（`/` Dashboard, `/annotator` 占位）
- [x] 命令行参数支持（--host, --port）

---

## 项目结构

```
refiner/
├── DOC/
│   └── refiner_implementation_status.md    # 本文档
├── anno_refiner_app/
│   ├── __init__.py
│   ├── main.py                             # 应用入口
│   └── src/
│       ├── __init__.py
│       ├── models.py                       # 数据模型定义
│       ├── state.py                        # 全局状态管理
│       ├── core/
│       │   ├── __init__.py
│       │   ├── yolo_utils.py               # YOLO 格式转换工具
│       │   ├── file_manager.py             # 文件备份/临时文件管理
│       │   └── analyzer.py                 # Cleanlab 分析器
│       └── ui/
│           ├── __init__.py
│           └── page_dashboard.py           # Dashboard 页面
├── TEST_stage_1_to_3/
│   ├── test_yolo_utils.py                  # yolo_utils 单元测试
│   ├── test_file_manager.py                # file_manager 单元测试
│   ├── test_analyzer.py                    # analyzer 集成测试
│   ├── test_full_workflow.py               # 完整工作流测试
│   ├── visualizations/                     # 问题样本可视化输出
│   └── DOC/
│       └── stage_1_to_3_summary.md         # 阶段1-3技术文档
└── 需求与规划_v1.md                         # 需求规格文档
```

---

## 启动方式

```bash
cd anno_refiner_app
python main.py --port 8088
```

然后在浏览器访问 `http://localhost:8088`（或通过端口转发访问）。

---

## 测试数据与结果

### 测试数据路径

| 数据类型 | 路径 |
|---------|------|
| Images | `/home/yangxinyu/Test/Data/internalVideos_fireRelated_annotatedFrames` |
| GT Labels | `/home/yangxinyu/Test/Data/internalVideos_fireRelated_keyFrameAnnotations_before` |
| Pred Labels | `/home/yangxinyu/Test/Data/internalVideos_fireRelated_annotatedFrames_predictions` |

### 测试结果统计

| 指标 | 数值 |
|-----|------|
| 图片总数 | 22,797 |
| GT 标签文件数 | 23,854 |
| Pred 标签文件数 | 22,148 |
| 检测到的类别数 | 2 |
| Overlooked 问题数 | 0（模型预测与 GT 匹配良好）|
| Swapped 问题数 | ~1000（TopK=1000 时）|
| Bad Located 问题数 | ~1000（TopK=1000 时）|

---

## 数据格式说明

### 目录结构

```
{root}/
├── {category}/
│   ├── {video}/
│   │   ├── frame_00025.jpg
│   │   ├── frame_00025.txt
│   │   └── ...
│   └── ...
└── ...
```

### YOLO 标签格式

**GT 格式**（无置信度）:
```
class_id center_x center_y width height
0 0.413644 0.451847 0.028454 0.051783
```

**Pred 格式**（有置信度）:
```
class_id center_x center_y width height confidence
0 0.413615 0.451451 0.029636 0.056187 0.793027
```

---

## Cleanlab 分析机制

### 分数含义

Cleanlab 为每个框计算质量分数（0-1），**分数越低表示问题越严重**：

- **Overlooked 分数**: 针对预测框，低分表示该预测框很可能是 GT 漏标的目标
- **Swap 分数**: 针对 GT 框，低分表示该 GT 框的类别很可能标错
- **Bad Located 分数**: 针对 GT 框，低分表示该 GT 框的位置很可能不准

### TopK 策略

当前实现采用 TopK 策略（而非固定阈值），取分数最低的 K 个样本作为最可疑的问题样本。原因：
1. 不同数据集的分数分布不同，固定阈值不通用
2. TopK 让用户可以控制审核工作量

---

## 待开发功能

### 阶段五：标注组件开发

- [ ] `InteractiveAnnotator` 基础版: 显示、选中
- [ ] 框的移动与调整大小
- [ ] 框的创建与删除
- [ ] 类别修改
- [ ] 撤销/重做功能

### 阶段六：页面B - Annotator

- [ ] 图片导航 (Previous/Next + 键盘快捷键)
- [ ] 显示选项控制
- [ ] 自动保存逻辑
- [ ] "Go Back Analysis" 确认流程

### 阶段七：集成测试与优化

- [ ] 端到端测试完整工作流
- [ ] 性能优化（大量图片加载）
- [ ] 错误处理与用户提示
- [ ] 样式美化

---

## 使用示例

### 命令行启动

```bash
cd /home/yangxinyu/Test/Projects/refiner/anno_refiner_app
python main.py --port 8088
```

### Python API 调用

```python
from anno_refiner_app.src.core.analyzer import CleanlabAnalyzer
from anno_refiner_app.src.models import IssueType

# 创建分析器
analyzer = CleanlabAnalyzer(
    images_path="/path/to/images",
    pred_labels_path="/path/to/predictions",
    gt_labels_path="/path/to/gt_labels",
    progress_callback=lambda msg, pct: print(f"[{pct*100:.0f}%] {msg}")
)

# 准备数据
analyzer.prepare_data()

# 执行分析
results = analyzer.analyze(top_k=10)

# 访问结果
for issue in results[IssueType.BAD_LOCATED]:
    print(f"{issue.image_path}: score={issue.score:.4f}, box={issue.box_index}")
```

---

## 变更日志

### 2026-01-15

- 初始化项目结构
- 完成阶段一至三的实现
- 通过全部单元测试和集成测试
- 生成问题样本可视化
- 创建本状态追踪文档
- 完成阶段四 Dashboard 页面开发
- 实现点击即时可视化功能
- 优化布局：三列固定高度问题列表 + 可视化面板
- 修复 backup 默认值问题（改为默认关闭）
- 添加 TopK 刷新按钮
