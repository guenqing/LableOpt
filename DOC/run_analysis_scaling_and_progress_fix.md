# RUN ANALYSIS 大目录卡死/断线与进度修复说明

## 1. 问题现象与触发条件

当 `Images/Labels` 目录规模很大（例如 Images 150w+）时：

- 前端容易出现 `Connection lost Trying to reconnect ...`
- 后端日志可见分析中途报错并中止，典型错误为：
  - `RuntimeError: The client this element belongs to has been deleted.`
- `RUN ANALYSIS` 显示的 remaining 数量异常偏大（例如 1499443），与预期“应该接近 GT 标注数量（例如 43388）”不一致。

## 2. 根因分析（对应旧实现）

### 2.1 样本枚举逻辑导致 remaining 过大

旧逻辑在 `CleanlabAnalyzer.prepare_data()` 中：

- 先对 **Images Path** 扫描全部图片（`collect_image_paths(images_path)`）
- 再仅按 `Output Path / Human Verified Path` 做“是否已处理”的跳过
- **不会**对 `GT Labels Path`、`Pred Labels Path` 做交集过滤
- 缺失 GT/Pred 标签时，转换阶段会把缺失当作“空标注/空预测”继续参与分析

因此当 Images Path 很大时，remaining 会接近 Images 总量（减去少量已处理样本）。

### 2.2 前端断线与后端 RuntimeError

旧实现里存在两类容易导致断线/崩溃的点：

- `RUN ANALYSIS` 点击后同步执行了重量级扫描（例如 `validate_paths()` 的 rglob 计数、以及为 Output 目录结构扫全量 Images），会阻塞事件循环，触发前端断线
- 分析在后台线程执行时，`progress_callback` 直接写 NiceGUI 的 UI 元素；一旦客户端断线，元素被销毁，再更新就会抛 `The client this element belongs to has been deleted`，从而让分析直接失败

## 3. 修复内容（当前实现）

### 3.1 RUN ANALYSIS 实际处理样本（新逻辑）

现在 `RUN ANALYSIS` 的样本集合为：

- **以 GT Labels 为基准**：遍历 `GT Labels Path` 下的 `*.txt`（排除 `*_tmp.txt`）
- 对每个样本 key（相对路径去掉后缀）同时要求：
  - **Pred label 存在**：`Pred Labels Path/{key}.txt` 存在
  - **Image 存在**：`Images Path/{key}.{jpg|jpeg|png|bmp}` 存在（自动探测扩展名）
- 然后再排除已处理样本：
  - 若 `Output Path` 或 `Human Verified Path` 中已存在 `{stem}.txt` 或 `{stem}_tmp.txt`，则跳过

这就等价于你描述的“**Images ∩ GT ∩ Pred 且不在 Output/HumanVerified**”，并且会使待处理数量上限接近 `gt_count`（例如 43388），不再出现 1499443 这种量级。

对应代码：`anno_refiner_app/src/core/analyzer.py::CleanlabAnalyzer.prepare_data()`

### 3.2 进度显示与断线容错

- `progress_callback` 不再直接更新 UI 元素
- Dashboard 的 `_update_progress()` 现在只更新全局状态：
  - `app_state.analysis_message`
  - `app_state.analysis_progress`
- UI 侧使用 `ui.timer` 周期性从 `app_state` 拉取并刷新进度条/按钮状态

即使前端断线，后台继续跑也不会因为 UI 元素被销毁而抛异常导致分析失败；重连后也能继续看到进度/结果。

对应代码：`anno_refiner_app/src/ui/page_dashboard.py`

### 3.3 避免点击 RUN ANALYSIS 时阻塞（减少断线概率）

- 点击 `RUN ANALYSIS` 不再做全量目录计数（不调用 `validate_paths()` 进行 rglob 统计）
- 不再为 Output 目录结构扫描全量 Images（移除 `collect_image_paths` + `ensure_output_structure` 的大扫描）
- 只做轻量的 `Path.exists()` 校验与创建 Output 根目录

## 4. 你关心的两个问题：结论

### 4.1 现在 RUN ANALYSIS 实际处理哪些样本？

如 3.1 所述：**以 GT label 文件为基准**，做 `Images ∩ GT ∩ Pred`，再排除 Output/HumanVerified 中已存在的样本。

### 4.2 为什么之前会是 1499443 remaining？

因为之前是“扫 Images 全量（1503103）→ 仅跳过 Output/HumanVerified（3660）→ 剩余 1499443”，并没有按 GT/Pred 做交集过滤。

## 5. 测试与回归

新增并运行的关键单元测试：

- `TEST_run_analysis_sample_selection_and_progress/test_analyzer_sample_selection.py`
  - 验证分析样本仅来自 Images∩GT∩Pred
  - 验证 Output 中已存在标签会被跳过
- `TEST_base_dir_and_parse_progress/test_intersection_filter.py`
  - 验证 label key 收集与过滤逻辑

命令示例：

```bash
python -m unittest discover -s TEST_run_analysis_sample_selection_and_progress -v
python -m unittest discover -s TEST_base_dir_and_parse_progress -v
```

## 6. 关键文件一览

- `anno_refiner_app/src/core/analyzer.py`
  - 样本交集枚举 + 跳过逻辑
- `anno_refiner_app/src/core/yolo_utils.py`
  - `collect_label_keys`
  - `filter_image_paths_by_label_keys`
  - `find_image_rel_path_for_key`
- `anno_refiner_app/src/ui/page_dashboard.py`
  - 进度状态写入 `app_state` + `ui.timer` 拉取刷新
  - 去除点击时的大扫描，降低断线概率

