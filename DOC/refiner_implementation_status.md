# Refiner 系统技术文档

> 最后更新: 2026-02-04
> 
> 最新更新: Extend GT to Next 增加子开关 Prefer Previous on Overlap（同类组替代 + 重叠采纳前一帧）；支持关闭 Extend 主开关即时回滚当前帧 Extend 结果（使用独立备份文件，不占用 *_tmp.txt）；无极缩放（1x-20x任意倍数，0.01步长）；自动聚焦算法优化；标签渲染智能合并与防重叠；默认启用自动保存和保存未修改；DLL加载问题修复；YOLO坐标转换精度改进；允许10%边界外滚动；新增内部标注一致性检测工具（findInconsistentAnno_internal.py）。

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

**新增分支：跳过 Cleanlab 直接标注（Direct Annotation）**：
- 不运行 `RUN ANALYSIS` 也可从 Dashboard 直接进入 Annotator
- 队列样本按 `Images ∩ (GT if set) ∩ (Pred if set)` 收集，并排除 Output/HumanVerified 中已存在的样本

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
│       │   ├── path_utils.py      # Base Dir 相对路径解析
│       │   ├── analyzer.py        # Cleanlab 分析器（支持样本过滤）
│       │   └── findInconsistentAnno_internal.py  # 内部标注一致性检测工具
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
    base_dir: str = "/home/yangxinyu/Test/Data/"
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
- `base_dir` 用于将 Dashboard 中输入的相对路径解析为绝对路径（可通过启动参数 `--base-dir` 覆盖）

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
    """像素坐标 (x1, y1, x2, y2) → YOLO 归一化坐标
    
    注意: 保留完整精度，不进行舍入操作
    """
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h
```

**坐标精度改进**:
- `pixel_to_yolo` 函数移除了坐标舍入操作，保留完整精度
- YOLO标签读取时放宽坐标范围验证，允许边界外的浮点误差
- 提高了坐标转换的精确性，避免因舍入造成的框位置偏移

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

#### 样本枚举辅助（交集过滤）

为避免在大数据集下扫描全量 Images，`yolo_utils.py` 提供了按“label key”枚举样本的辅助函数：

- `collect_label_keys(labels_dir, exclude_tmp=True)`: 收集标签文件对应的 key（相对路径去后缀），可排除 `*_tmp.txt`
- `find_image_rel_path_for_key(images_dir, key)`: 根据 key 在 Images Path 下自动探测图片扩展名（`.jpg/.jpeg/.png/.bmp`），返回实际存在的相对路径
- `filter_image_paths_by_label_keys(image_rel_paths, gt_keys, pred_keys)`: 仅保留同时存在 GT/Pred 的图片路径（用于小规模或已有 image 列表的场景）

### 2.4 路径解析工具 (`src/core/path_utils.py`)

- `resolve_with_base_dir(base_dir, user_path)`: 将 UI 输入的相对路径基于 `base_dir` 解析为绝对路径；绝对路径原样返回；空输入返回空字符串

### 2.5 文件管理 (`src/core/file_manager.py`)

**核心功能**:
- `save_tmp_annotation()`: 保存临时标注到Output Path
- `confirm_changes()`: 确认或丢弃更改（操作Output Path）
- `get_tmp_files()`: 获取Output Path中的临时文件列表
- `validate_output_path()`: 验证Output Path并检查与GT/Pred路径的冲突
- `ensure_output_structure()`: 批量创建Output子目录结构（大数据集下不建议在RUN ANALYSIS前全量扫描创建，现已改为按需创建）
- `should_skip_sample()`: 检查样本是否已在Output Path或Human Verified Path中存在
- `collect_annotation_image_paths()`: 直接标注模式的样本队列构建（`Images ∩ (GT if set) ∩ (Pred if set)`，再排除已处理）
- `estimate_pending_analysis_samples()`: 估算分析待处理样本（`Images ∩ GT ∩ Pred - processed`）
- `parse_data_for_dashboard()`: Dashboard 侧手动解析入口，按 label key 统计 valid/missing_img，并按新口径计算 pending

**样本过滤逻辑**:
- RUN ANALYSIS样本由 `analyzer.prepare_data()` 先按 `GT Labels ∩ Pred Labels ∩ Images` 收集
- 之后自动跳过已在Output Path或Human Verified Annotation Path中存在标签的样本
- 检查 `.txt` 和 `_tmp.txt` 文件
- 如果Human Verified Path为空，则只检查Output Path
- 直接标注模式样本按 `Images ∩ (GT if set) ∩ (Pred if set)` 收集，并同样跳过 Output/HumanVerified 中已存在的样本

### 3. Cleanlab 分析器 (`src/core/analyzer.py`)

#### 分析流程

```python
class CleanlabAnalyzer:
    def __init__(self, ..., output_path: str = "", human_verified_path: str = ""):
        self.output_path = output_path
        self.human_verified_path = human_verified_path
    
    def prepare_data(self):
        """准备数据: 格式转换"""
        # 1. 收集候选样本（避免扫描全量 Images）
        #    - 遍历 GT Labels 下的 *.txt（排除 *_tmp.txt）
        #    - 要求 Pred Labels/{key}.txt 存在
        #    - 要求 Images/{key}.{jpg|jpeg|png|bmp} 存在（自动探测扩展名）
        #    - 跳过 Output/HumanVerified 中已存在 .txt 或 _tmp.txt 的样本
        filtered_paths = ...  # 遍历GT label files，检查Pred存在、匹配Image存在（自动探测扩展名），再用should_skip_sample过滤
        
        # 2. 转换 GT 标签
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
- **Base Dir**: 相对路径解析基准目录（UI输入相对路径时会与Base Dir拼接；可通过启动参数 `--base-dir` 设置）
- **Images Path** (必填): 图片目录
- **Output Path** (必填): 输出路径，默认 `/home/yangxinyu/Test/Data/internalVideos_fireRelated_keyFrameAnnotations_verifying`
  - 系统会检查是否与GT/Pred路径相同，相同则警告但不禁止
  - RUN ANALYSIS时仅确保Output Path根目录存在；子目录在保存标注时按需创建（避免大目录预扫描）
- **GT Labels Path** (可选): GT标签目录
- **Pred Labels Path** (可选): 预测标签目录
- **Human Verified Annotation Path** (可选): 人工验证路径，可为空
- **Classes File** (可选): 类别映射文件
- **Parse Data**: 手动触发路径解析与统计（位于 `RUN ANALYSIS` 上方）
- **Pending Samples**: 待处理样本数（醒目显示）
  - 定义：`Images ∩ (GT if set) ∩ (Pred if set) - Output - Human Verified`
  - Output 以 `{stem}.txt` 与 `{stem}_tmp.txt` 归一，Human Verified 仅计 `{stem}.txt`

**路径解析**:
- 路径输入变化时仅同步配置并清空上次统计；不再自动解析
- 点击 `Parse Data` 后才解析：每个输入框右侧显示 `OK/Not found`，输入框下方显示 `valid/missing_img`
- Images-only 模式全量扫描 Images，其它模式按 label key 探测图片存在并缓存结果
- RUN ANALYSIS点击时只做轻量 `Path.exists()` 校验，并强制要求 Images+GT+Pred 同时存在，否则弹窗提示并拒绝执行

#### 运行分析与进度（断线重连安全）

- 分析进度由后端写入全局状态（`app_state.analysis_message` / `app_state.analysis_progress`）
- UI端使用 `ui.timer` 轮询刷新进度条与按钮状态
- 客户端断线/重连不会导致后台因更新已销毁UI元素而抛异常，从而避免分析中途失败

#### 选择逻辑

进入 Annotator 存在两种路径：

- **分析模式（analysis_complete=True）**：用户通过勾选 "Overlooked/Swapped/Bad Located" 复选框选择问题类型
  - 只勾选某一类 → 只包含该类型样本
  - 勾选多类 → 合并去重后进入
  - 未勾选任何类型 → 提示至少选择一个类型
- **直接标注模式（analysis_complete=False）**：忽略问题类型复选框，按 `Images ∩ (GT if set) ∩ (Pred if set)` 构建队列并排除 Output/HumanVerified 已处理样本（`IssueType.DIRECT`）

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
│ [Image]  │[Thumb]   │ GT #1    │  [ ] Auto Focus            │
│          │          │ Pred #0  │  [ ] Show GT               │
│          │          │ ...      │  [ ] Show Pred             │
│          │          │ ...      │                             │
│          │          │          │ Zoom Controls               │
│ [Prev]   │          │ [eye]    │  [1x] [+][-][Reset]         │
│ [Next]   │          │          │  [Slider]                    │
│          │          │          │                             │
│          │          │          │ Box Actions                 │
│          │          │          │  [Swap Editable]            │
│          │          │          │  [Clear Editable]           │
│          │          │          │  [Activate Reference]       │
│          │          │          │                             │
│          │          │          │ Save Controls                │
│          │          │          │  [x] Extend GT to Next (默认)│
│          │          │          │  [ ] Prefer Previous on Overlap│
│          │          │          │  [x] Auto Save (默认启用)   │
│          │          │          │  [x] Save Unmodified (默认) │
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
    # 扫描 Output 下的 *_tmp.txt（离线线程执行，避免阻塞 UI）
    tmp_files = get_tmp_files(app_state.config.output_path)
    
    if not tmp_files:
        # 无修改，直接返回
        ui.navigate.to('/')
    else:
        # 显示确认对话框
        # Yes: 覆盖原文件 (confirm_changes(keep_changes=True))
        # No: 删除临时文件 (confirm_changes(keep_changes=False))
        # Cancel: 停留在当前页面
```

**自动保存逻辑**:
- Auto Save 默认开启，切换图片会自动保存
- Save Unmodified 默认开启，即使未修改也会保存当前可编辑框

**Extend GT to Next（跨帧复用可编辑框）**:
- UI 位置：Annotator 右侧 `Save Controls` 区域，默认开启
- 快捷键：`y` 切换开/关
- 行为：进入下一帧时，先将上一帧的可编辑框复制到下一帧（作为 GT，可编辑），再做去重：如果复制来的框与下一帧已有可编辑框“同类别且 IoU > 0.2”，则丢弃复制框（优先保留下一帧已有框）；其它复制框会被追加到下一帧
- 坐标策略：复制使用 YOLO 归一化坐标（相对坐标），在下一帧按目标分辨率还原为像素框，适配分辨率变化
- Auto Save 兼容：当 Extend 在下一帧实际追加了复制框时，会将该帧标记为 `modified`，确保在关闭 `Save Unmodified` 时切图仍会触发自动保存

**Prefer Previous on Overlap（Extend 子开关）**:
- 仅在 `Extend GT to Next` 开启时可勾选并生效
- 含义：允许“同类组替代”，并在重叠时优先采纳前一帧（删除后一帧重叠的可编辑框，保留复制框）
- 同类组替代规则：
  - 前一帧 `0` 或 `2` 可替代后一帧的 `0/2`
  - 前一帧 `1` 或 `3` 可替代后一帧的 `1/3`
  - 其它类别默认只替代自身类别
- 重叠判定：与后一帧**可编辑框**满足 `IoU > 0.2` 且同替代组时触发替代；Pred 默认不可编辑，因此默认不会被替代删除

**关闭 Extend 回滚（当前帧）**:
- 如果某帧是通过 Extend 生成/修改的，系统会在应用 Extend 前，将该帧原始 `GT/Pred` 框备份到 Output 下的独立文件：
  - `Output Path/.refiner_extend_backups/<relative_image_path>.json`
  - 文件名与 `*_tmp.txt` 无冲突，不会影响保存/确认流程
- 当你停留在该帧并关闭 `Extend GT to Next` 主开关时，会立刻从备份恢复该帧原标注（不影响你把恢复后的标注继续用于后续帧的 Extend）

**近期交互/兼容性修复**:
- `Go Back to Analysis` 的确认对话框按钮回调改为直接 `await`（避免 `create_task` 丢失 client 上下文导致需要二次点击返回）
- `_load_boxes()` 在 GT/Pred 路径为空时不再从 CWD 兜底读取标签文件（直接标注模式下 GT/Pred 允许为空）

### 7. InteractiveAnnotator 组件 (`src/ui/components.py`)

#### 交互功能

**鼠标操作**:
- 点击框: 选中框 (显示 8 个控制点)
- 拖拽框: 移动框
- 拖拽控制点: 调整大小 (4 角 + 4 边)
- 空白区域拖拽: 创建新框 (支持任意缩放级别，不再限制在1x)

**键盘快捷键**:
```
┌─────────────────────────────────────────────────────────┐
│  导航                                                    │
│  [ / ]       上一张/下一张图片                         │
│  ·            循环选中可编辑框                          │
│                                                          │
│  缩放                                                    │
│  = / +        放大一级                                   │
│  -            缩小一级                                   │
│  0            重置到 1x                                  │
│                                                          │
│  显示控制                                                │
│  q            切换 Show GT                               │
│  w            切换 Show Pred                             │
│                                                          │
│  框操作                                                  │
│  e            交换可编辑状态 (Swap Editable)             │
│  r            清除可编辑框 (Clear Editable)              │
│  t            激活参照框 (Activate Reference)            │
│                                                          │
│  编辑                                                    │
│  Del/Backspace  删除选中框                              │
│  Arrow Keys    移动选中框 (1px, Shift=10px)             │
│  1/2/3/4       修改类别为 0/1/2/3                       │
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

#### 放大状态下创建框

**修复内容**:
- 移除了创建框功能在放大状态下的限制
- 现在可以在任意缩放级别（1x-20x）下通过鼠标拖拽创建标注框
- `interactive_image` 组件提供的 `image_x/image_y` 已经是图像坐标系坐标（不受CSS transform影响），可以直接使用

**实现细节**:
- 在 `_on_mouse_down` 方法中，移除了 `elif self.zoom <= 1.0:` 的条件限制
- 空白区域拖拽时，无论当前缩放级别如何，都可以创建新框

#### 标签渲染优化

**智能合并与防重叠**:
- 当多个框重叠时，标签文本会智能合并显示，避免遮挡
- 标签位置自动调整，防止与相邻框的标签重叠
- 提升密集标注场景下的可读性

#### 视图边界处理

**允许10%边界外滚动**:
- 视图平移时允许超出图像边界10%的范围
- 便于查看和编辑位于图像边缘的标注框
- 改善边界区域的操作体验

#### 显示选项同步

**功能**:
- 通过快捷键 `q`/`w` 切换 Show GT/Pred 时，UI上的checkbox会自动同步更新
- 通过UI checkbox切换时，显示状态也会同步更新

**实现**:
- 添加 `on_display_change` 回调机制
- 快捷键切换时调用回调更新checkbox状态
- UI checkbox变化时调用 `set_display_options` 更新显示状态

#### 新框类别默认值

**修复内容**:
- 在加载新图片时，自动重置 `current_class` 为 0
- 确保新创建的框默认使用类别 0，而不是之前设置的类别

**实现**:
- 在 `load_image()` 方法中，加载图片后重置 `current_class = 0`
- 避免因之前按过类别快捷键（1/2/3/4）而影响新图片的默认类别

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
# 无极缩放: 1x 到 20x 任意倍数，步长 0.01
# 不再使用固定级别列表，支持连续缩放

# 快捷键
# = / + : 放大
# -     : 缩小
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

#### 自动聚焦功能

系统在加载图片和标注框后，可根据 `Auto Focus` 勾选状态决定是否自动聚焦或保持 1x 显示。

**实现原理**:

```python
def auto_focus_boxes(self) -> None:
    """根据GT和Pred框的并集，自动设置视野中心和放大倍率"""
    # 1. 计算所有框的并集
    min_x = min(所有框的x)
    min_y = min(所有框的y)
    max_x = max(所有框的x+w)
    max_y = max(所有框的y+h)
    
    # 2. 计算框的尺寸
    box_width = max_x - min_x
    box_height = max_y - min_y
    
    # 3. 计算最优放大倍率（向下取整）
    # 在zoom级别Z下，可见区域: view_width/(Z*scale_1x) x view_height/(Z*scale_1x)
    # 需要: visible_width >= box_width 且 visible_height >= box_height
    zoom_x = view_width / (box_width * scale_1x)
    zoom_y = view_height / (box_height * scale_1x)
    optimal_zoom = floor(min(zoom_x, zoom_y))  # 向下取整
    
    # 4. 计算视野中心（框的中心）
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # 5. 设置pan使中心点位于视野中心
    visible_w = view_width / (zoom * scale_1x)
    visible_h = view_height / (zoom * scale_1x)
    pan_x = center_x - visible_w / 2
    pan_y = center_y - visible_h / 2
```

**触发时机**:
- 在 `_load_current_image()` 中，加载完图片和框后触发
- `Auto Focus` 勾选时执行自动聚焦，取消勾选时重置到 1x

**优势**:
- 无需手动缩放和定位，提高标注效率
- 自动计算最优倍率，确保所有框都在视野中
- 视野中心自动对齐到框的中心，方便查看和编辑

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
- Output Path目录结构通过“按相对路径保存 + 按需创建子目录”与GT Labels Path保持一致（不在RUN ANALYSIS前全量预创建）
- 支持Human Verified Annotation Path标记已人工验证的样本

**样本过滤**:
- RUN ANALYSIS样本按 `GT Labels ∩ Pred Labels ∩ Images` 收集，并自动跳过已在Output Path或Human Verified Path中存在的样本（检查 `.txt` / `_tmp.txt`）
- 避免重复处理已修正的样本，提高效率

**选择逻辑**:
- 分析模式：Dashboard 中通过复选框选择问题类型（Overlooked/Swapped/Bad Located），进入标注队列前支持多选与去重
- 直接标注模式：忽略问题类型复选框，按 `Images ∩ (GT if set) ∩ (Pred if set) - processed` 构建队列

---

## 启动与使用

### 启动应用

```bash
cd anno_refiner_app
python main.py --port 8088 --base-dir /path/to/data
```

访问 `http://localhost:8088`

### 基本使用流程

```
1. 配置路径
   ├─ Images Path（必填）: 图片目录
   ├─ Output Path（必填）: 输出路径（修正后的标注保存路径）
   ├─ GT Labels Path（可选）: GT 标签目录
   ├─ Pred Labels Path（可选）: 预测标签目录
   ├─ Human Verified Annotation Path（可选）: 人工验证路径
   └─ Classes File（可选）: 类别映射文件

2A. 运行分析（可选）
   ├─ 前提：Images + GT + Pred 均已配置且存在
   ├─ 分析样本按 Images ∩ GT ∩ Pred 收集，并跳过 Output/Human Verified 中已存在的样本
   └─ 点击 "RUN ANALYSIS" → 等待完成

2B. 直接进入标注（可选，跳过分析）
   ├─ 前提：Images + Output 已配置
   ├─ 队列样本按 Images ∩ (GT if set) ∩ (Pred if set) 收集，并跳过 Output/Human Verified 中已存在的样本
   └─ 点击 "Go to Annotation Tool" → 直接进入 Annotator

3. 查看问题
   ├─ 三列问题列表显示
   ├─ 点击问题项查看可视化
   └─ 选择问题类型 (复选框) - 可多选，用于过滤标注队列

4. 进入标注
   ├─ 分析模式：设置 TopK 值 + 勾选问题类型 → 点击 "Go to Annotation Tool"
   └─ 直接标注模式：无需问题类型选择，点击 "Go to Annotation Tool"

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
│       │   ├── path_utils.py  # Base Dir 相对路径解析
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
