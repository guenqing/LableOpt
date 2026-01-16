# Refiner 系统开发状态追踪

> 最后更新: 2026-01-16

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
| 阶段五 | 标注组件开发 | 已完成 | 2026-01-15 |
| 阶段六 | 页面B - Annotator | 已完成 | 2026-01-15 |
| 阶段七 | 集成测试与优化 | 已完成 | 2026-01-16 |
| 阶段八 | 性能优化 | 已完成 | 2026-01-16 |

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
- [x] `prepare_cleanlab_labels()`: 将 GT 标签转换为 Cleanlab 格式（支持自动并行/串行切换）
- [x] `prepare_cleanlab_predictions()`: 将预测标签转换为 Cleanlab 格式（支持自动并行/串行切换）
- [x] `count_classes()`: 统计类别数量（支持自动并行/串行切换）

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
- [x] 使用 `auxiliary_inputs` 优化分析阶段性能（减少重复计算）

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

### 阶段五：标注组件开发

#### `TEST_stage_5/interactive_annotator.py`

核心交互式标注组件 `InteractiveAnnotator`，基于 NiceGUI `ui.interactive_image` 实现：

**基础功能**：
- [x] 图片加载与显示
- [x] GT 框渲染（绿色实线）
- [x] Pred 框渲染（蓝色虚线）
- [x] 选中框高亮（红色，显示 8 个控制点）

**鼠标交互**：
- [x] 点击选中框
- [x] 拖拽移动框
- [x] 拖拽控制点调整大小（4 角 + 4 边）
- [x] 空白区域拖拽创建新框

**键盘快捷键**：
- [x] `Tab`: 循环选中 GT 框
- [x] `Delete/Backspace`: 删除选中框
- [x] `Arrow Keys`: 移动选中框（1px，Shift=10px）
- [x] `q/w/e/r`: 修改类别为 0/1/2/3
- [x] `a/s/d/f`: 边框向外扩展（Shift 反向）
- [x] `z/x/c/v`: 角向外扩展（Shift 反向）
- [x] `Ctrl+Z`: 撤销
- [x] `Ctrl+Y`: 重做

**缩放与平移**：
- [x] 缩放级别 1x-10x
- [x] 滚动条快速平移
- [ ] ~~拖拽平移~~（已禁用，有坐标计算问题）
- [ ] 缩放焦点定位（BUG-001：选中框后缩放不能正确居中）

**约束与历史**：
- [x] 框约束在图片边界内
- [x] 最小框尺寸 5x5 像素
- [x] 撤销/重做历史栈（最多 50 步）

#### `TEST_stage_5/test_annotator.py`

独立测试页面，用于验证组件功能：
- 随机加载 10 张测试图片
- 完整的控制面板（显示选项、类别选择、缩放控制）
- 快捷键参考提示

### 阶段六：页面B - Annotator

#### `src/ui/components.py`

从 `TEST_stage_5/interactive_annotator.py` 复制并适配导入路径：
- [x] InteractiveAnnotator 组件完整集成
- [x] 修改导入为相对路径 (`from ..models import BBox, BoxSource`)

#### `src/ui/page_annotator.py`

Annotator 页面实现 `AnnotatorPage` 类：

**标题栏**：
- [x] 图片路径显示
- [x] 进度显示 `(15/47)`
- [x] 问题类型与分数显示

**标注区域**：
- [x] InteractiveAnnotator 组件集成（900x600 固定尺寸）
- [x] 导航按钮 `[<- Prev]` / `[Next ->]`

**缩略图**：
- [x] Navigator 缩略图区域，显示当前图片的缩略图以及视野框（蓝色半透明框）
- [x] 缩略图区域点击可以跳转到标注区域相应位置
- [x] 缩略图区域拖动可以移动标注区域视野框

**控制面板**：
- [x] Display Options（Show GT / Show Pred 复选框）
- [x] Zoom Controls（缩放倍数、+/-按钮、滑动条、重置）
- [x] Current Class（下拉选择 0/1/2/3）
- [x] Save Controls（Auto Save 复选框、Save 按钮）
- [x] Go Back to Analysis 按钮
- [x] 键盘快捷键提示展开面板

**导航功能**：
- [x] `[` / `]` 键：上一张/下一张图片
- [x] 边界处理（第一张禁用 Prev，最后一张禁用 Next）

**保存逻辑**：
- [x] 手动保存：保存到 `{stem}_tmp.txt`
- [x] Auto Save：切换图片时自动保存
- [x] 加载时优先读取 tmp 文件

**返回确认流程**：
- [x] 检测 tmp 文件存在性
- [x] 确认对话框（Yes-Keep / No-Discard / Cancel）
- [x] Yes：覆盖原文件
- [x] No：删除 tmp 文件
- [x] Cancel：停留在当前页面
- [x] 无修改时直接返回

#### `main.py`

- [x] 更新 `/annotator` 路由指向 `page_annotator.create_annotator()`

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
│           ├── components.py               # InteractiveAnnotator 组件
│           ├── page_dashboard.py           # Dashboard 页面 (Page A)
│           └── page_annotator.py           # Annotator 页面 (Page B)
├── TEST_stage_1_to_3/
│   ├── test_yolo_utils.py                  # yolo_utils 单元测试
│   ├── test_file_manager.py                # file_manager 单元测试
│   ├── test_analyzer.py                    # analyzer 集成测试
│   ├── test_full_workflow.py               # 完整工作流测试
│   ├── visualizations/                     # 问题样本可视化输出
│   └── DOC/
│       └── stage_1_to_3_summary.md         # 阶段1-3技术文档
├── TEST_stage_5/
│   ├── interactive_annotator.py            # InteractiveAnnotator 组件（源）
│   ├── test_annotator.py                   # 测试页面入口
│   ├── test_unit.py                        # 单元测试
│   └── DOC/
│       └── stage_5_summary.md              # 阶段5技术文档
├── TEST_page_b_annotator/
│   ├── test_annotator_page.py              # 页面B单元测试
│   ├── test_integration.py                 # 集成测试
│   └── DOC/
│       └── page_b_summary.md               # 阶段6技术文档
├── TEST_stage_7/                           # 阶段7测试
│   ├── test_models.py                      # BBox 模型测试
│   ├── test_annotator_logic.py             # 一键操作逻辑测试
│   └── test_e2e_runner.py                  # 端到端测试运行器
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

## 已完成功能：阶段七

### 阶段七：集成测试与易用性优化

#### 7.1 新功能：标注框列表面板

在 Navigator 缩略图右侧、Control Panel 左侧增加 Box List 面板：
- [x] 显示所有 GT 和 Pred 框（序号、类别、来源）
- [x] 点击列表项可选中标注区域对应框（仅可编辑框）
- [x] 每个框右侧有"眼睛"图标，控制单框显示/隐藏
- [x] 数据模型 `BBox` 新增 `visible` 字段

#### 7.2 UI 优化：类别标签放大

- [x] 类别标签字体从 12px 放大到 16px
- [x] 添加白色半透明背景（85%不透明度），增强对比度

#### 7.3 新功能：一键处理（GT/Pred 可编辑状态切换）

数据模型 `BBox` 新增 `editable` 字段，支持灵活整合 GT 和 Pred 数据：
- [x] `editable=True`: 可编辑（实线边框）
- [x] `editable=False`: 仅参照（虚线边框）
- [x] 默认：GT 可编辑，Pred 不可编辑

新增三个一键操作按钮：
- [x] **Swap Editable**: 切换所有框的可编辑状态（GT<->Pred 角色互换）
- [x] **Clear Editable**: 删除全部可编辑框
- [x] **Activate Reference**: 将参照框改为可编辑（合并 Pred 到 GT）
- [x] 三个操作均支持撤销/重做

#### 7.4 已知问题修复

- [x] BUG-001：缩放焦点定位（选中框后缩放正确居中到选中框）
- [x] 拖拽平移已禁用（有坐标计算问题，保持现状，使用滚动条/minimap）

#### 7.5 端到端测试

- [x] 完整工作流测试（页面A -> 分析 -> 页面B -> 编辑 -> 保存 -> 返回）
- [x] 单元测试：BBox 模型测试、一键操作逻辑测试
- [x] 撤销/重做测试

#### 7.6 保存逻辑增强

- [x] 保存时只导出 `editable=True` 的框（支持 Pred 框激活后保存）
- [x] HistoryState 保存完整状态（包括 pred_boxes、visible、editable）

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

### InteractiveAnnotator 组件使用

```python
from nicegui import ui
from TEST_stage_5.interactive_annotator import InteractiveAnnotator
from anno_refiner_app.src.models import BBox, BoxSource

# 创建组件
def on_boxes_changed(boxes):
    print(f"GT boxes updated: {len(boxes)} boxes")

annotator = InteractiveAnnotator(
    on_change=on_boxes_changed,
    on_zoom_change=lambda z: print(f"Zoom: {z}x")
)

# 在 UI 容器中创建
container = ui.element('div')
annotator.create_ui(container, fixed_width=900, fixed_height=600)

# 加载图片和标注
annotator.load_image('/path/to/image.jpg')
annotator.load_boxes(gt_boxes, pred_boxes)

# 控制显示
annotator.set_display_options(show_gt=True, show_pred=True)

# 缩放控制
annotator.set_zoom(2.0)  # 设置 2x 缩放
annotator.zoom_in()      # 放大
annotator.zoom_out()     # 缩小
annotator.reset_zoom()   # 重置为 1x

# 获取编辑后的 GT 框
edited_boxes = annotator.get_gt_boxes()
```

### 阶段五测试页面启动

```bash
cd /home/yangxinyu/Test/Projects/refiner/TEST_stage_5
/home/yangxinyu/Test/anaconda3/envs/parse/bin/python test_annotator.py --port 8089
```

然后访问 `http://localhost:8089`

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
- **完成阶段五 InteractiveAnnotator 组件开发**
  - 基于 NiceGUI ui.interactive_image 的交互式标注组件
  - 支持鼠标拖拽（移动、调整大小、创建框）
  - 支持 8 方向控制点（4 角 + 4 边）
  - 丰富的键盘快捷键（类别修改、边框微调、角微调）
  - 缩放（1x-10x）与平移功能
  - 撤销/重做历史栈
  - 独立测试页面验证通过
- **完成阶段六 页面B - Annotator 开发**
  - 复制 InteractiveAnnotator 到 `src/ui/components.py`
  - 创建 `src/ui/page_annotator.py` 实现完整页面
  - 标题栏显示路径、进度、问题类型、分数
  - 控制面板：Display Options、Zoom、Class、Save
  - 图片导航：`[`/`]` 键 + Prev/Next 按钮
  - 保存逻辑：手动保存、Auto Save、tmp 文件管理
  - 返回确认：检测修改、三选项对话框（Keep/Discard/Cancel）
  - 更新 main.py 路由
  - 单元测试和集成测试通过
- **页面B 优化与修复**
  - 移除左上角后退按钮，强制用户通过 "Go Back to Analysis" 确认保存
  - 修正样本数量显示：分析结果按类别独立显示数量
  - 修正 TopK 逻辑：进入 annotation 时对每个选中类型取 TopK 再合并去重
  - 修复确认对话框按钮文字居中对齐
  - 禁用放大状态下的拖动平移（有坐标计算问题），改用滚动条平移

### 2026-01-16

- **完成阶段七：集成测试与易用性优化**
  - 新增 Box List 面板：显示所有框、点击选中、眼睛图标控制可见性
  - 放大类别标签字体（12px -> 16px）并添加白色背景
  - 新增 `visible` 和 `editable` 字段到 BBox 数据模型
  - 新增三个一键操作按钮：Swap Editable、Clear Editable、Activate Reference
  - 修复 BUG-001：缩放焦点定位问题（选中框后缩放正确居中）
  - 增强 HistoryState：保存 pred_boxes 和所有框属性
  - 增强保存逻辑：只保存 editable=True 的框
  - 创建单元测试：test_models.py、test_annotator_logic.py
  - 创建端到端测试运行器：test_e2e_runner.py
  - 编写阶段7技术文档：DOC/stage_7_summary.md
- **完成阶段八：性能优化**
  - 阶段一：使用 `auxiliary_inputs` 优化 Cleanlab 分析阶段（加速 ~2x）
  - 阶段二：实现基于 ProcessPoolExecutor 的数据准备并行化
    - 自动选择策略：小样本（<10k）使用串行，大样本（≥10k）使用并行
    - 最优配置：32 进程，chunksize = len(tasks) // (workers * 4)
    - 预期加速：100 万样本时 3-6x 加速（从 ~7.7 分钟降至 ~1.3-2.6 分钟）
  - 添加详细的性能日志记录（各步骤耗时统计）
  - 编写性能分析文档：DOC/run_analysis_performance_analysis.md
  - 编写并行化分析文档：DOC/parallelization_analysis.md
  - 编写并行化成功原因分析：DOC/parallelization_success_explanation.md
