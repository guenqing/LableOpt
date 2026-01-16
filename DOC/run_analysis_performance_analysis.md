# RUN ANALYSIS 性能分析与加速方案

> 最后更新: 2026-01-16

## 一、完整计算流程

### 1.1 数据准备阶段 (`prepare_data()`)

#### 步骤 1: 收集图片路径 (5% 进度)
- **函数**: `collect_image_paths(images_dir)`
- **操作**: 递归遍历目录结构，收集所有图片文件路径
- **耗时**: 文件系统 I/O，相对较快
- **并行化潜力**: ⭐ 低（文件系统遍历本身难以并行，且耗时占比小）

#### 步骤 2: 转换 GT 标签 (10% 进度)
- **函数**: `prepare_cleanlab_labels(images_dir, gt_labels_dir, image_rel_paths)`
- **操作**: 
  - 对每张图片：
    1. 读取图片尺寸 (`get_image_size`) - I/O 操作
    2. 读取对应的 GT 标签文件 (`read_yolo_label`) - I/O 操作
    3. 转换 YOLO 格式到 Cleanlab 格式 - CPU 计算
- **耗时**: 主要来自大量图片的串行 I/O 和格式转换
- **并行化潜力**: ⭐⭐⭐⭐⭐ 高（每张图片处理完全独立）

#### 步骤 3: 统计类别数 (20% 进度)
- **函数**: `count_classes(gt_labels_dir, pred_labels_dir, image_rel_paths)`
- **操作**: 
  - 遍历所有标签文件，读取类别 ID
  - 找出最大类别 ID，计算 `num_classes = max(class_id) + 1`
- **耗时**: 文件 I/O 为主
- **并行化潜力**: ⭐⭐⭐⭐ 中高（每张图片的类别统计独立）

#### 步骤 4: 转换 Pred 标签 (30% 进度)
- **函数**: `prepare_cleanlab_predictions(images_dir, pred_labels_dir, image_paths, num_classes)`
- **操作**: 
  - 对每张图片：
    1. 读取图片尺寸 (`get_image_size`) - I/O 操作
    2. 读取对应的 Pred 标签文件 (`read_yolo_label`) - I/O 操作
    3. 按类别组织预测框 - CPU 计算
    4. 转换为 Cleanlab 格式 - CPU 计算
- **耗时**: 主要来自大量图片的串行 I/O 和格式转换
- **并行化潜力**: ⭐⭐⭐⭐⭐ 高（每张图片处理完全独立）

### 1.2 分析阶段 (`analyze()`)

#### 步骤 5: 计算 Overlooked 分数 (50% 进度)
- **函数**: `compute_overlooked_box_scores(labels, predictions)`
- **操作**: Cleanlab 内部计算每张图片的预测框被漏标的可能性
- **耗时**: 计算密集型，涉及 IoU 计算、相似度矩阵等
- **并行化潜力**: ⭐⭐ 低（Cleanlab 内部实现，无原生并行参数）

#### 步骤 6: 计算 Swap 分数 (60% 进度)
- **函数**: `compute_swap_box_scores(labels, predictions)`
- **操作**: Cleanlab 内部计算每张图片的 GT 框类别错误的可能性
- **耗时**: 计算密集型
- **并行化潜力**: ⭐⭐ 低（Cleanlab 内部实现，无原生并行参数）

#### 步骤 7: 计算 Bad Location 分数 (70% 进度)
- **函数**: `compute_badloc_box_scores(labels, predictions)`
- **操作**: Cleanlab 内部计算每张图片的 GT 框位置不准的可能性
- **耗时**: 计算密集型
- **并行化潜力**: ⭐⭐ 低（Cleanlab 内部实现，无原生并行参数）

#### 步骤 8: 排序和筛选结果 (80% 进度)
- **操作**: 
  - 对每张图片的分数进行过滤（去除 NaN）
  - 找出每张图片的最低分数
  - 按分数排序，取 TopK
- **耗时**: 相对较快，主要是 Python 列表操作
- **并行化潜力**: ⭐⭐⭐ 中（可以并行处理三种问题类型）

## 二、Cleanlab 原生 API 加速选项

### 2.1 `auxiliary_inputs` 参数优化

根据 Cleanlab 文档，三个 score 计算函数都支持 `auxiliary_inputs` 参数，可以：
- **减少重复计算**: 三个函数都需要计算相似度矩阵、分离标签和预测等，使用 `auxiliary_inputs` 可以预先计算一次，然后共享
- **预期加速**: 约 20-30% 的加速（避免重复计算相似度矩阵）

**实现方式**:
```python
# 预先计算辅助输入
auxiliary_inputs = _get_valid_inputs_for_compute_scores(
    alpha=None, labels=self.labels, predictions=self.predictions
)

# 三个函数共享辅助输入
overlooked_scores = compute_overlooked_box_scores(
    auxiliary_inputs=auxiliary_inputs
)
swap_scores = compute_swap_box_scores(
    auxiliary_inputs=auxiliary_inputs
)
badloc_scores = compute_badloc_box_scores(
    auxiliary_inputs=auxiliary_inputs
)
```

### 2.2 无原生并行化支持

经过查询 Cleanlab 文档，发现：
- `compute_overlooked_box_scores`、`compute_swap_box_scores`、`compute_badloc_box_scores` 这三个函数**没有直接的并行化参数**
- Cleanlab 内部在某些函数（如 `_calculate_ap_per_class`）使用了 `multiprocessing.Pool`，但这是内部实现，不对外暴露

## 三、自定义加速方案

### 3.1 数据准备阶段并行化

#### 方案 A: 并行化 GT 标签转换
- **实现**: 使用 `multiprocessing.Pool` 或 `concurrent.futures.ProcessPoolExecutor`
- **并行粒度**: 按图片并行处理
- **预期加速**: 2-4x（取决于 CPU 核心数和 I/O 瓶颈）
- **注意事项**: 
  - 需要处理图片读取的异常
  - 需要保持结果顺序（使用 `map` 而非 `imap_unordered`）

#### 方案 B: 并行化 Pred 标签转换
- **实现**: 同上
- **预期加速**: 2-4x

#### 方案 C: 合并 GT 和 Pred 转换
- **实现**: 在一次并行处理中同时转换 GT 和 Pred，减少图片尺寸读取次数
- **预期加速**: 额外 10-20%（减少重复的图片尺寸读取）

#### 方案 D: 并行化类别统计
- **实现**: 并行读取标签文件统计类别
- **预期加速**: 1.5-2x（I/O 密集型，收益相对较小）

### 3.2 分析阶段优化

#### 方案 E: 使用 `auxiliary_inputs` 减少重复计算
- **实现**: 预先计算辅助输入，三个函数共享
- **预期加速**: 20-30%
- **优先级**: ⭐⭐⭐⭐⭐ 高（实现简单，收益明显）

#### 方案 F: 并行化三种分数计算（如果 Cleanlab 支持）
- **现状**: Cleanlab 函数内部实现，无法直接并行化
- **替代方案**: 如果数据量极大，可以考虑数据分片，但需要验证 Cleanlab 是否支持

#### 方案 G: 并行化结果排序和筛选
- **实现**: 三种问题类型的排序可以并行进行
- **预期加速**: 很小（这部分耗时占比低）

### 3.3 综合优化策略

**优先级排序**:
1. ⭐⭐⭐⭐⭐ **方案 E**: 使用 `auxiliary_inputs`（实现简单，收益明显）
2. ⭐⭐⭐⭐ **方案 A + B**: 并行化 GT 和 Pred 标签转换（收益大，但需要处理进程间通信）
3. ⭐⭐⭐ **方案 C**: 合并 GT 和 Pred 转换（减少重复 I/O）
4. ⭐⭐ **方案 D**: 并行化类别统计（收益较小）

## 四、实施建议

### 4.1 第一阶段：快速收益（方案 E）

**目标**: 使用 `auxiliary_inputs` 减少重复计算

**修改点**:
- `src/core/analyzer.py` 的 `analyze()` 方法
- 需要导入 `_get_valid_inputs_for_compute_scores`（可能需要检查 cleanlab 版本和 API）

**预期效果**: 20-30% 加速，几乎无副作用

### 4.2 第二阶段：数据准备并行化（方案 A + B + C）

**目标**: 并行化数据准备阶段

**修改点**:
- `src/core/yolo_utils.py` 的 `prepare_cleanlab_labels()` 和 `prepare_cleanlab_predictions()`
- 或者创建新的并行版本函数

**实现方式**:
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def _process_single_image_gt_pred(args):
    """处理单张图片的 GT 和 Pred 转换"""
    # 返回 (gt_label_dict, pred_array, image_path)
    pass

def prepare_cleanlab_data_parallel(...):
    """并行版本的数据准备"""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 并行处理
        results = list(executor.map(_process_single_image_gt_pred, image_paths))
    # 分离 GT 和 Pred
    return labels, predictions, image_paths
```

**注意事项**:
- 需要处理进程间通信（序列化/反序列化）
- 需要处理异常（某些图片可能无法读取）
- 需要保持结果顺序
- 可能需要调整进度回调机制

**预期效果**: 2-4x 加速（取决于 CPU 核心数）

### 4.3 测试策略

1. **基准测试**: 记录当前实现的耗时
2. **分阶段测试**: 每个优化方案单独测试，验证加速效果
3. **集成测试**: 确保优化后功能正确性不变
4. **性能测试**: 在不同数据规模下测试（小、中、大）

## 五、预期总体加速效果

假设当前总耗时 100 单位：

| 阶段 | 当前耗时 | 优化后耗时 | 加速比 |
|------|---------|-----------|--------|
| 数据准备 | 40 | 10-15 | 2.5-4x |
| 分析计算 | 55 | 40-45 | 1.2-1.4x |
| 结果排序 | 5 | 5 | 1x |
| **总计** | **100** | **55-65** | **1.5-1.8x** |

**保守估计**: 总体加速 **1.5-2x**

**理想情况**（多核 CPU + 快速 I/O）: 总体加速 **2-3x**

## 六、实施检查清单

- [ ] 阶段一：实现 `auxiliary_inputs` 优化
  - [ ] 检查 cleanlab 版本和 API 兼容性
  - [ ] 修改 `analyzer.py` 的 `analyze()` 方法
  - [ ] 测试功能正确性
  - [ ] 性能测试（对比优化前后）

- [ ] 阶段二：实现数据准备并行化
  - [ ] 创建并行版本的转换函数
  - [ ] 处理异常和进度回调
  - [ ] 测试功能正确性
  - [ ] 性能测试（不同数据规模）
  - [ ] 优化进程数配置（可配置参数）

- [ ] 阶段三：集成和优化
  - [ ] 合并优化方案
  - [ ] 端到端测试
  - [ ] 文档更新
