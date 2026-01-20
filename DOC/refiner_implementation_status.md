# Refiner 系统技术文档

> 最后更新: 2026-01-16
> 
> 最新更新: 添加Output Path功能，修复选择逻辑，优化文件保存机制

## 系统功能与逻辑概述

Refiner 是一个基于 Cleanlab 的 YOLO 目标检测标签修正工具，通过 Web UI 自动定位可能有误的标注样本，支持人工审核和修正。

### 核心工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      Refiner 工作流程                            │
└─────────────────────────────────────────────────────────────────┘

1. 数据准备阶段
   ┌──────────────┐
   │ 输入路径配置 │ → 验证路径有效性 → 收集图片和标签文件
   └──────────────┘

2. Cleanlab 分析阶段
   ┌─────────────────────────────────────────────┐
   │ 格式转换 (YOLO → Cleanlab)                  │
   │  ├─ GT 标签转换                              │
   │  ├─ Pred 标签转换                            │
   │  └─ 类别统计                                 │
   │                                             │
   │ 问题检测 (三种类型)                          │
   │  ├─ Overlooked: 模型预测但 GT 未标注         │
   │  ├─ Swapped: GT 类别与预测类别不一致         │
   │  └─ Bad Located: GT 框与预测框 IoU 低        │
   │                                             │
   │ 分数排序 → TopK 筛选                        │
   └─────────────────────────────────────────────┘

3. 可视化与筛选阶段 (Dashboard)
   ┌─────────────────────────────────────────────┐
   │ 问题列表展示 (三列)                         │
   │  ├─ Overlooked (橙色)                       │
   │  ├─ Swapped (红色)                          │
   │  └─ Bad Located (紫色)                       │
   │                                             │
   │ 点击问题项 → 即时可视化                      │
   │  ├─ GT 框 (绿色实线)                        │
   │  ├─ Pred 框 (蓝色虚线)                      │
   │  └─ 问题框 (红色高亮)                       │
   │                                             │
   │ 选择问题类型 → 构建标注队列                  │
   └─────────────────────────────────────────────┘

4. 标注修正阶段 (Annotator)
   ┌─────────────────────────────────────────────┐
   │ 交互式标注编辑                               │
   │  ├─ 鼠标操作: 拖拽移动/调整大小/创建框       │
   │  ├─ 键盘快捷键: 类别修改/微调/导航          │
   │  └─ 一键操作: 交换可编辑状态/清除/激活        │
   │                                             │
   │ 保存机制                                     │
   │  ├─ 临时文件: {stem}_tmp.txt                │
   │  ├─ 自动保存: 切换图片时自动保存             │
   │  └─ 确认保存: 返回时确认覆盖或丢弃           │
   └─────────────────────────────────────────────┘
```

### 三类标签问题检测原理

| 问题类型 | 检测原理 | 代码位置 |
|---------|---------|---------|
| **Overlooked（漏标）** | 模型高置信度预测了框，但 GT 无对应标注 | `analyzer.py:165-182` |
| **Swapped（类别错误）** | GT 框类别与模型预测类别不一致 | `analyzer.py:186-203` |
| **Bad Located（框不准）** | GT 框与模型预测框的 IoU 较低 | `analyzer.py:207-224` |

---

## 系统架构

### 项目结构

```
refiner/
├── anno_refiner_app/
│   ├── main.py                    # 应用入口，路由配置
│   └── src/
│       ├── models.py              # 数据模型 (BBox, IssueItem, etc.)
│       ├── state.py               # 全局状态管理 (AppState)
│       ├── core/
│       │   ├── yolo_utils.py      # YOLO 格式转换工具
│       │   ├── file_manager.py   # 文件备份/临时文件管理/输出路径管理
│       │   └── analyzer.py        # Cleanlab 分析器（支持样本过滤）
│       └── ui/
│           ├── components.py      # InteractiveAnnotator 组件
│           ├── page_dashboard.py  # Dashboard 页面
│           └── page_annotator.py  # Annotator 页面
└── DOC/
    └── refiner_implementation_status.md
```

### 技术栈

- **前端框架**: NiceGUI (Python Web UI)
- **问题检测**: Cleanlab object_detection
- **数据处理**: NumPy, PIL
- **并行处理**: ProcessPoolExecutor (自动切换串行/并行)

---

## 核心模块详解

### 1. 数据模型 (`src/models.py`)

系统定义了三个核心数据模型：

```python
@dataclass
class BBox:
    """UI 层边界框"""
    x: float                    # 左上角 x (像素)
    y: float                    # 左上角 y (像素)
    w: float                    # 宽度 (像素)
    h: float                    # 高度 (像素)
    class_id: int               # 类别 ID
    source: BoxSource           # 来源 (GT/PRED)
    id: str                     # 唯一标识符
    selected: bool = False      # 是否选中
    visible: bool = True        # 是否可见
    editable: bool = True       # 是否可编辑

@dataclass
class IssueItem:
    """问题样本"""
    image_path: str             # 相对路径
    issue_type: IssueType       # 问题类型
    score: float                # 严重程度分数 (越低越严重)
    box_index: Optional[int]    # 问题框索引

@dataclass
class SessionConfig:
    """会话配置"""
    images_path: str = ""
    pred_labels_path: str = ""
    gt_labels_path: str = ""
    output_path: str = ""       # 输出路径（必填）：修正后的标注保存路径
    human_verified_path: str = ""  # 人工验证路径（可选）：已人工验证的标注路径
    classes_file: str = ""
    top_k: int = 10
    backup_enabled: bool = False
```

**设计要点**:
- `editable` 字段支持 GT/Pred 角色互换，实现灵活的数据整合
- `visible` 字段支持单框显示/隐藏控制
- `source` 枚举区分 GT 和 Pred 框的来源
- `output_path` 用于保存修正后的标注，避免直接修改原始GT标签

### 2. YOLO 格式转换 (`src/core/yolo_utils.py`)

#### 坐标转换

```python
def yolo_to_pixel(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """YOLO 归一化坐标 → 像素坐标 (x1, y1, x2, y2)"""
    box_w = w * img_w
    box_h = h * img_h
    x1 = (cx * img_w) - (box_w / 2)
    y1 = (cy * img_h) - (box_h / 2)
    x2 = x1 + box_w
    y2 = y1 + box_h
    return x1, y1, x2, y2

def pixel_to_yolo(x1: float, y1: float, x2: float, y2: float,
                  img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """像素坐标 (x1, y1, x2, y2) → YOLO 归一化坐标"""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h
```

#### 并行处理策略

系统根据样本数量自动选择串行或并行处理：

```python
PARALLEL_THRESHOLD = 10000  # 阈值: 10k 样本

def prepare_cleanlab_labels(...):
    num_samples = len(image_rel_paths)
    
    if num_samples >= PARALLEL_THRESHOLD:
        # 并行处理 (ProcessPoolExecutor)
        num_workers = min(multiprocessing.cpu_count(), 32)
        chunksize = max(1, len(args_list) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker, args_list, chunksize=chunksize))
    else:
        # 串行处理 (小数据集)
        for rel_path in image_rel_paths:
            # 处理单个样本
```

**性能优化**:
- 小样本 (<10k): 串行处理，避免进程创建开销
- 大样本 (≥10k): 并行处理，最多 32 进程
- 预期加速: 100 万样本时 3-6x 加速

### 2.5 文件管理 (`src/core/file_manager.py`)

**核心功能**:
- `save_tmp_annotation()`: 保存临时标注到Output Path
- `confirm_changes()`: 确认或丢弃更改（操作Output Path）
- `get_tmp_files()`: 获取Output Path中的临时文件列表
- `validate_output_path()`: 验证Output Path并检查与GT/Pred路径的冲突
- `ensure_output_structure()`: 确保Output Path目录结构与GT Labels Path相同
- `should_skip_sample()`: 检查样本是否已在Output Path或Human Verified Path中存在

**样本过滤逻辑**:
- 在RUN ANALYSIS时，自动跳过已在Output Path或Human Verified Annotation Path中存在标签的样本
- 检查 `.txt` 和 `_tmp.txt` 文件
- 如果Human Verified Path为空，则只检查Output Path

### 3. Cleanlab 分析器 (`src/core/analyzer.py`)

#### 分析流程

```python
class CleanlabAnalyzer:
    def __init__(self, ..., output_path: str = "", human_verified_path: str = ""):
        self.output_path = output_path
        self.human_verified_path = human_verified_path
    
    def prepare_data(self):
        """准备数据: 格式转换"""
        # 1. 收集图片路径
        all_image_rel_paths = collect_image_paths(self.images_path)
        
        # 2. 过滤已处理的样本（跳过Output Path或Human Verified Path中已存在的样本）
        from .file_manager import should_skip_sample
        filtered_paths = [p for p in all_image_rel_paths 
                         if not should_skip_sample(str(p), self.output_path, self.human_verified_path)]
        
        # 3. 转换 GT 标签
        self.labels, self.image_paths = prepare_cleanlab_labels(..., filtered_paths)
        
        # 3. 统计类别数
        self.num_classes = count_classes(...)
        
        # 4. 转换 Pred 标签
        self.predictions = prepare_cleanlab_predictions(...)
    
    def analyze(self, top_k: int = 10):
        """执行分析并返回 TopK 问题样本"""
        # 1. 预计算辅助输入 (避免重复计算)
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(
            alpha=ALPHA,
            labels=self.labels,
            predictions=self.predictions
        )
        
        # 2. 计算三种问题分数
        overlooked_scores = compute_overlooked_box_scores(
            auxiliary_inputs=auxiliary_inputs
        )
        swap_scores = compute_swap_box_scores(auxiliary_inputs=auxiliary_inputs)
        badloc_scores = compute_badloc_box_scores(auxiliary_inputs=auxiliary_inputs)
        
        # 3. 过滤 NaN 值并排序
        # NaN 表示完美匹配，不报告为问题
        for img_path, scores in zip(self.image_paths, overlooked_scores):
            valid_mask = ~np.isnan(scores)
            if np.any(valid_mask):
                # 取最低分数 (最可疑)
                min_score = float(np.min(scores[valid_mask]))
                # 添加到结果列表
        
        # 4. 按分数排序，取 TopK
        results[IssueType.OVERLOOKED].sort(key=lambda x: x.score)
        return {type: items[:top_k] for type, items in results.items()}
```

**关键优化**:
- 使用 `auxiliary_inputs` 共享中间计算结果，加速 ~2x
- 过滤 NaN 值: 完美匹配的框不报告为问题
- TopK 策略: 取分数最低的 K 个样本，而非固定阈值

#### 分数含义

Cleanlab 为每个框计算质量分数 (0-1)，**分数越低表示问题越严重**:

```
┌─────────────────────────────────────────────────────────┐
│  Overlooked 分数                                         │
│  ─────────────────────────────────────────────────────  │
│  针对: 预测框                                            │
│  低分 → 该预测框很可能是 GT 漏标的目标                   │
│                                                          │
│  Swapped 分数                                            │
│  ─────────────────────────────────────────────────────  │
│  针对: GT 框                                             │
│  低分 → 该 GT 框的类别很可能标错                          │
│                                                          │
│  Bad Located 分数                                        │
│  ─────────────────────────────────────────────────────  │
│  针对: GT 框                                             │
│  低分 → 该 GT 框的位置很可能不准                          │
└─────────────────────────────────────────────────────────┘
```

### 4. 全局状态管理 (`src/state.py`)

系统使用全局单例管理应用状态：

```python
@dataclass
class AppState:
    config: SessionConfig              # 会话配置
    results: AnalysisResults           # 分析结果
    is_analyzing: bool                 # 分析状态
    annotation_queue: List[IssueItem]  # 标注队列
    current_annotation_index: int      # 当前标注索引
    
    def get_selected_issues(self, top_k: int = None) -> List[IssueItem]:
        """合并选中类型的问题 (去重)
        
        根据checkbox状态过滤，只返回选中类型的问题样本
        """
        seen_paths = set()
        merged = []
        
        # 对每个选中类型取 TopK，然后合并去重
        if self.selected_overlooked:
            items = self.results.overlooked[:top_k] if top_k else self.results.overlooked
            for item in items:
                if item.image_path not in seen_paths:
                    seen_paths.add(item.image_path)
                    merged.append(item)
        # ... 类似处理 swapped 和 bad_located
        
        return merged

# 全局实例
app_state = AppState()
```

**设计要点**:
- 去重逻辑: 同一图片可能出现在多种问题类型中，合并时去重
- TopK 应用: 进入标注前对每个类型取 TopK，再合并

### 5. Dashboard 页面 (`src/ui/page_dashboard.py`)

#### 路径配置

Dashboard配置面板包含以下路径输入：
- **Images Path**: 图片目录
- **GT Labels Path**: GT标签目录
- **Pred Labels Path**: 预测标签目录
- **Output Path** (必填): 输出路径，默认 `/home/yangxinyu/Test/Data/internalVideos_fireRelated_keyFrameAnnotations_verifying`
  - 系统会检查是否与GT/Pred路径相同，相同则警告但不禁止
  - 首次使用时自动创建目录结构（与GT Labels Path结构相同）
- **Human Verified Annotation Path** (可选): 人工验证路径，可为空
- **Classes File** (可选): 类别映射文件

**路径验证**:
- Output Path未设置时，RUN ANALYSIS会弹窗警告并拒绝执行
- 自动验证路径有效性并显示文件统计

#### 选择逻辑

用户可以通过勾选"Overlooked"、"Swapped"、"Bad Located"复选框来选择进入标注工具的问题类型：
- 只勾选"Overlooked" → 只包含Overlooked类型的样本
- 勾选多个类型 → 包含所有选中类型的样本（去重）
- 未勾选任何类型 → 提示用户至少选择一个类型

#### 布局结构

```
┌─────────────────────────────────────────────────────────────┐
│  Header: Annotation Refiner                                 │
├──────────┬──────────────────────────┬───────────────────────┤
│          │                          │                        │
│ Config   │  Issue Lists (3 columns) │  Visualization        │
│ Panel    │                          │  Panel                │
│          │  ┌──────┬──────┬──────┐ │                        │
│ - Paths  │  │Over- │Swap- │Bad   │ │  [Click issue to      │
│   Images │  │looked│ped   │Loc.  │ │   visualize]          │
│   GT     │  │      │      │      │ │                        │
│   Pred   │  │List  │List  │List  │ │                        │
│   Output │  └──────┴──────┴──────┘ │                        │
│   Human  │                          │                        │
│   Verified│                         │                        │
│ - Run    │  TopK: [10] [Refresh]    │                        │
│ - Backup │  [Go to Annotation Tool]  │                        │
└──────────┴──────────────────────────┴───────────────────────┘
```

#### 即时可视化

点击问题列表项时，系统动态生成可视化图片：

```python
def _generate_visualization(self, item: IssueItem) -> str:
    """生成可视化图片"""
    # 加载图片和标签
    img = Image.open(img_path)
    gt_boxes = read_yolo_label(gt_label_path, img_w, img_h, has_confidence=False)
    pred_boxes = read_yolo_label(pred_label_path, img_w, img_h, has_confidence=True)
    
    # 绘制
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img)
    
    # GT 框: 绿色实线，问题框红色高亮
    for i, box in enumerate(gt_boxes):
        is_issue = (i == item.box_index and 
                   item.issue_type in [IssueType.SWAPPED, IssueType.BAD_LOCATED])
        color = 'red' if is_issue else 'lime'
        # 绘制矩形...
    
    # Pred 框: 蓝色虚线，问题框橙色高亮
    for i, box in enumerate(pred_boxes):
        is_issue = (i == item.box_index and item.issue_type == IssueType.OVERLOOKED)
        color = 'orange' if is_issue else 'deepskyblue'
        # 绘制矩形...
    
    # 保存到临时文件
    temp_path = tempfile.mkstemp(suffix='.png', prefix='refiner_viz_')
    plt.savefig(temp_path, dpi=120)
    return temp_path
```

### 6. Annotator 页面 (`src/ui/page_annotator.py`)

#### 布局结构

```
┌─────────────────────────────────────────────────────────────┐
│  Header: image_path | (15/47) | Issue Type | Score          │
├──────────┬──────────┬──────────┬─────────────────────────────┤
│          │          │          │                             │
│ Viewer   │Navigator │Box List  │ Control Panel              │
│ (900x600)│(Minimap) │          │                             │
│          │          │ GT #0    │ Display Options            │
│ [Image]  │[Thumb]   │ GT #1    │  ☑ Show GT                 │
│          │          │ Pred #0  │  ☑ Show Pred               │
│          │          │ ...      │                             │
│          │          │          │ Zoom Controls               │
│ [Prev]   │          │ [👁]     │  [1x] [+][-][Reset]         │
│ [Next]   │          │          │  [Slider]                    │
│          │          │          │                             │
│          │          │          │ Box Actions                 │
│          │          │          │  [Swap Editable]            │
│          │          │          │  [Clear Editable]           │
│          │          │          │  [Activate Reference]       │
│          │          │          │                             │
│          │          │          │ Save Controls                │
│          │          │          │  ☑ Auto Save                │
│          │          │          │  [Save]                     │
│          │          │          │  [Go Back to Analysis]      │
└──────────┴──────────┴──────────┴─────────────────────────────┘
```

#### 保存机制

```python
def _on_save(self):
    """保存当前标注 (只保存 editable=True 的框)"""
    # 获取所有框
    all_boxes = self.annotator.get_all_boxes()
    
    # 过滤可编辑框 (支持 Pred 框激活后保存)
    boxes_dict = []
    for box in all_boxes:
        if getattr(box, 'editable', True):
            boxes_dict.append({
                'class_id': box.class_id,
                'bbox': [box.x, box.y, box.x + box.w, box.y + box.h]
            })
    
    # 保存到临时文件: {stem}_tmp.txt (保存到Output Path)
    save_tmp_annotation(
        app_state.config.output_path,
        item.image_path,
        boxes_dict,
        img_w, img_h
    )

def _on_back(self):
    """返回时确认保存"""
    tmp_files = get_tmp_files(app_state.config.gt_labels_path)
    
    if not tmp_files:
        # 无修改，直接返回
        ui.navigate.to('/')
    else:
        # 显示确认对话框
        # Yes: 覆盖原文件 (confirm_changes(keep_changes=True))
        # No: 删除临时文件 (confirm_changes(keep_changes=False))
        # Cancel: 停留在当前页面
```

### 7. InteractiveAnnotator 组件 (`src/ui/components.py`)

#### 交互功能

**鼠标操作**:
- 点击框: 选中框 (显示 8 个控制点)
- 拖拽框: 移动框
- 拖拽控制点: 调整大小 (4 角 + 4 边)
- 空白区域拖拽: 创建新框

**键盘快捷键**:
```
┌─────────────────────────────────────────────────────────┐
│  导航                                                    │
│  [ / ]       上一张/下一张图片                         │
│  Tab          循环选中 GT 框                             │
│                                                          │
│  缩放                                                    │
│  = / +        放大一级                                   │
│  -            缩小一级                                   │
│  0            重置到 1x                                  │
│                                                          │
│  编辑                                                    │
│  Del/Backspace  删除选中框                              │
│  Arrow Keys    移动选中框 (1px, Shift=10px)             │
│  q/w/e/r       修改类别为 0/1/2/3                       │
│  a/s/d/f       边框向外扩展 (Shift 反向)                 │
│  z/x/c/v       角向外扩展 (Shift 反向)                  │
│  Ctrl+Z/Y      撤销/重做                                │
└─────────────────────────────────────────────────────────┘
```

#### 一键操作

```python
def swap_editable(self):
    """交换所有框的可编辑状态 (GT ↔ Pred 角色互换)"""
    for box in self.gt_boxes + self.pred_boxes:
        box.editable = not box.editable
    self._save_history()

def clear_editable(self):
    """删除所有可编辑框"""
    self.gt_boxes = [b for b in self.gt_boxes if not b.editable]
    self.pred_boxes = [b for b in self.pred_boxes if not b.editable]
    self._save_history()

def activate_reference(self):
    """将参照框改为可编辑 (合并 Pred 到 GT)"""
    for box in self.pred_boxes:
        if not box.editable:
            box.editable = True
    self._save_history()
```

**使用场景**:
- **Swap Editable**: 当 Pred 框质量更好时，交换角色进行编辑
- **Clear Editable**: 快速删除所有 GT 框，重新标注
- **Activate Reference**: 将 Pred 框合并到 GT，用于补充漏标

---

## 数据流程

### 输入数据格式

#### 目录结构

```
{root}/
├── {category}/
│   ├── {video}/
│   │   ├── frame_00025.jpg
│   │   ├── frame_00025.txt      # GT 标签
│   │   └── ...
│   └── ...
└── ...
```

#### YOLO 标签格式

**GT 格式** (无置信度):
```
class_id center_x center_y width height
0 0.413644 0.451847 0.028454 0.051783
```

**Pred 格式** (有置信度):
```
class_id center_x center_y width height confidence
0 0.413615 0.451451 0.029636 0.056187 0.793027
```

### 数据转换流程

```
┌─────────────────────────────────────────────────────────┐
│  1. YOLO 格式 (归一化坐标)                               │
│     class_id cx cy w h [confidence]                     │
│                                                          │
│  2. 像素坐标转换                                         │
│     yolo_to_pixel() → (x1, y1, x2, y2)                  │
│                                                          │
│  3. Cleanlab 格式                                        │
│     GT: {'bboxes': np.array, 'labels': np.array}        │
│     Pred: np.array of arrays (按类别分组)                │
│                                                          │
│  4. 分析结果                                             │
│     IssueItem(image_path, issue_type, score, box_index)  │
│                                                          │
│  5. UI 层 BBox                                           │
│     BBox(x, y, w, h, class_id, source, editable, ...)    │
│                                                          │
│  6. 保存回 YOLO 格式                                     │
│     pixel_to_yolo() → (cx, cy, w, h)                    │
└─────────────────────────────────────────────────────────┘
```

### 临时文件机制

```
原始GT文件: GT_Labels_Path/category/video/frame_00025.txt
输出路径: Output_Path/category/video/frame_00025.txt
临时文件: Output_Path/category/video/frame_00025_tmp.txt

加载逻辑:
  1. 优先读取 Output_Path 中的 _tmp.txt (如果存在)
  2. 其次读取 Output_Path 中的 .txt (如果存在)
  3. 最后回退到原始 GT_Labels_Path 中的 .txt

保存逻辑:
  1. 编辑后保存到 Output_Path 的 _tmp.txt
  2. 返回时确认:
     - Yes: _tmp.txt → .txt (覆盖Output_Path中的文件)
     - No: 删除 _tmp.txt
```

**关键变更**:
- 所有修正后的标注都保存到 `Output Path`，不再直接修改 `GT Labels Path`
- `Output Path` 必须设置，否则无法运行分析
- 支持 `Human Verified Annotation Path`（可选），用于标记已人工验证的样本

---

## 关键技术与概念

### 1. Cleanlab 问题检测原理

Cleanlab 基于置信学习 (Confident Learning) 理论，通过比较模型预测与标注的一致性来识别标签问题。

#### Overlooked 检测

```
模型预测: [Pred Box A, Pred Box B, ...]
GT 标注:  [GT Box A]

问题: Pred Box B 在 GT 中无对应框
      → 可能是 GT 漏标了该目标
      → Overlooked 分数低
```

#### Swapped 检测

```
GT Box A: class_id=0 (类别 0)
Pred Box A: class_id=1 (类别 1), IoU 高

问题: 位置匹配但类别不一致
      → 可能是 GT 类别标错
      → Swap 分数低
```

#### Bad Located 检测

```
GT Box A: (x1, y1, x2, y2)
Pred Box A: (x1', y1', x2', y2'), IoU < 阈值

问题: 类别匹配但位置偏差大
      → 可能是 GT 框位置不准
      → Bad Located 分数低
```

### 2. TopK 策略 vs 固定阈值

**为什么使用 TopK**:

```
固定阈值的问题:
  - 不同数据集分数分布不同
  - 难以设置通用阈值
  - 可能漏掉重要问题或包含过多噪声

TopK 策略的优势:
  - 自适应: 取最可疑的 K 个样本
  - 可控: 用户控制审核工作量
  - 通用: 适用于各种数据集
```

**实现**:
```python
# 对每种问题类型分别取 TopK
overlooked_items.sort(key=lambda x: x.score)
results[IssueType.OVERLOOKED] = overlooked_items[:top_k]

# 合并时去重 (同一图片可能出现在多种类型中)
def get_selected_issues(self, top_k: int = None):
    seen_paths = set()
    merged = []
    # ... 合并逻辑
```

### 3. 并行处理优化

#### 自动策略选择

```python
PARALLEL_THRESHOLD = 10000

if num_samples >= PARALLEL_THRESHOLD:
    # 并行: 大数据集
    num_workers = min(multiprocessing.cpu_count(), 32)
else:
    # 串行: 小数据集 (避免进程创建开销)
```

#### 进程池配置

```python
# 最优配置
chunksize = max(1, len(tasks) // (workers * 4))

# 原因: 
# - chunksize 太小: 进程间通信开销大
# - chunksize 太大: 负载不均衡
# - 经验值: tasks / (workers * 4)
```

#### 性能提升

```
数据集规模: 100 万样本
串行处理: ~7.7 分钟
并行处理: ~1.3-2.6 分钟 (32 进程)
加速比: 3-6x
```

### 4. 撤销/重做机制

```python
@dataclass
class HistoryState:
    """历史状态快照"""
    gt_boxes: List[BBox]
    pred_boxes: List[BBox]
    selected_id: Optional[str] = None

class InteractiveAnnotator:
    history: List[HistoryState] = []
    history_index: int = -1
    MAX_HISTORY = 50
    
    def _save_history(self):
        """保存当前状态到历史栈"""
        state = HistoryState(
            gt_boxes=deepcopy(self.gt_boxes),
            pred_boxes=deepcopy(self.pred_boxes),
            selected_id=self.selected_box_id
        )
        
        # 删除当前索引之后的历史 (分支操作)
        self.history = self.history[:self.history_index + 1]
        
        # 添加新状态
        self.history.append(state)
        self.history_index = len(self.history) - 1
        
        # 限制历史长度
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self):
        """撤销"""
        if self.history_index > 0:
            self.history_index -= 1
            self._restore_state(self.history[self.history_index])
    
    def redo(self):
        """重做"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self._restore_state(self.history[self.history_index])
```

**设计要点**:
- 深拷贝: 避免状态被意外修改
- 分支处理: 新操作时删除后续历史
- 长度限制: 最多保存 50 步历史

### 5. 可编辑状态设计

`editable` 字段实现了 GT/Pred 框的灵活整合:

```python
# 默认状态
GT boxes:   editable=True  (可编辑，实线边框)
Pred boxes:  editable=False   (仅参照，虚线边框)

# Swap Editable 后
GT boxes:   editable=False   (变为参照)
Pred boxes:  editable=True    (变为可编辑)

# 使用场景
1. Pred 框质量更好 → Swap → 编辑 Pred 框
2. 重新标注 → Clear Editable → 创建新框
3. 补充漏标 → Activate Reference → Pred 框合并到 GT
```

### 6. 缩放与平移实现

#### 缩放级别与快捷键

```python
# 缩放级别: 1x 到 10x
ZOOM_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 快捷键
# = / + : 放大一级
# -     : 缩小一级
# 0     : 重置到 1x
```

#### 坐标系统

系统使用统一的图像坐标系统：
- `pan_x`, `pan_y`: 视野左上角在图像坐标系中的位置（像素）
- 所有坐标转换基于图像尺寸和显示缩放比例 `scale_1x`
- Viewer 和 Navigator 使用相同的坐标映射关系

#### 缩放焦点定位

```python
def set_zoom(self, zoom: float, focus_point: tuple = None):
    """设置缩放级别
    
    焦点优先级:
    1. 如果选中了框，以框中心为焦点
    2. 否则，以当前视野中心为焦点
    """
    # 计算焦点位置（图像坐标）
    if focus_point is None:
        focus_point = self._get_zoom_focus_point()
    
    # 设置缩放并调整 pan，使焦点保持在视野中心
    self.zoom = new_zoom
    visible_w = self.view_width / (self.zoom * scale_1x)
    visible_h = self.view_height / (self.zoom * scale_1x)
    self.pan_x = focus_x - visible_w / 2
    self.pan_y = focus_y - visible_h / 2
```

#### Navigator (小地图)

- **功能**: 点击 Navigator 任意位置，Viewer 快速定位到该位置
- **实现**: 使用 `interactive_image` 组件，坐标直接映射到图像坐标
- **特点**: 不显示视野框，简化交互，避免坐标换算问题

### 7. 性能优化技巧

#### auxiliary_inputs 优化

```python
# 优化前: 三次独立计算，重复计算相似度矩阵
overlooked_scores = compute_overlooked_box_scores(...)
swap_scores = compute_swap_box_scores(...)
badloc_scores = compute_badloc_box_scores(...)

# 优化后: 预计算共享输入
auxiliary_inputs = _get_valid_inputs_for_compute_scores(...)
overlooked_scores = compute_overlooked_box_scores(auxiliary_inputs=auxiliary_inputs)
swap_scores = compute_swap_box_scores(auxiliary_inputs=auxiliary_inputs)
badloc_scores = compute_badloc_box_scores(auxiliary_inputs=auxiliary_inputs)

# 加速: ~2x
```

#### NaN 值过滤

```python
# NaN 表示完美匹配，不报告为问题
valid_mask = ~np.isnan(scores)
if np.any(valid_mask):
    valid_scores = scores[valid_mask]
    min_score = float(np.min(valid_scores))
    # 只处理有效分数
```

### 8. Output Path 与样本过滤

**Output Path机制**:
- 所有修正后的标注保存到Output Path，不直接修改GT Labels Path
- Output Path目录结构自动与GT Labels Path保持一致
- 支持Human Verified Annotation Path标记已人工验证的样本

**样本过滤**:
- RUN ANALYSIS时自动跳过已在Output Path或Human Verified Path中存在的样本
- 避免重复处理已修正的样本，提高效率

**选择逻辑**:
- Dashboard中通过复选框选择问题类型（Overlooked/Swapped/Bad Located）
- 进入标注工具时只包含选中类型的问题样本
- 支持多选，自动去重

---

## 启动与使用

### 启动应用

```bash
cd anno_refiner_app
python main.py --port 8088
```

访问 `http://localhost:8088`

### 基本使用流程

```
1. 配置路径
   ├─ Images Path: 图片目录
   ├─ GT Labels Path: GT 标签目录
   ├─ Pred Labels Path: 预测标签目录
   ├─ Output Path: 输出路径（必填），修正后的标注保存路径
   ├─ Human Verified Annotation Path: 人工验证路径（可选）
   └─ Classes File (可选): 类别映射文件

2. 运行分析
   ├─ 系统自动创建 Output Path 目录结构
   ├─ 自动跳过已在 Output Path 或 Human Verified Path 中存在的样本
   └─ 点击 "RUN ANALYSIS" → 等待完成

3. 查看问题
   ├─ 三列问题列表显示
   ├─ 点击问题项查看可视化
   └─ 选择问题类型 (复选框) - 可多选，用于过滤标注队列

4. 进入标注
   ├─ 设置 TopK 值
   ├─ 勾选需要标注的问题类型（Overlooked/Swapped/Bad Located）
   └─ 点击 "Go to Annotation Tool" - 只包含选中类型的问题

5. 编辑标注
   ├─ 使用鼠标/键盘编辑框
   ├─ 使用一键操作快速处理
   └─ 保存修改 (手动或自动) - 保存到 Output Path

6. 确认保存
   └─ 返回时选择 Keep/Discard - 操作 Output Path 中的文件
```

---

## 项目结构总结

```
refiner/
├── anno_refiner_app/          # 主应用
│   ├── main.py                # 入口: 路由配置
│   └── src/
│       ├── models.py          # 数据模型
│       ├── state.py           # 状态管理
│       ├── core/              # 核心模块
│       │   ├── yolo_utils.py  # YOLO 转换
│       │   ├── file_manager.py # 文件管理
│       │   └── analyzer.py    # Cleanlab 分析
│       └── ui/                # 用户界面
│           ├── components.py  # 标注组件
│           ├── page_dashboard.py # Dashboard
│           └── page_annotator.py # Annotator
└── DOC/                       # 文档
    └── refiner_implementation_status.md
```

---

## 参考资源

- **Cleanlab**: https://github.com/cleanlab/cleanlab
- **NiceGUI**: https://nicegui.io/
- **YOLO 格式**: https://docs.ultralytics.com/datasets/
