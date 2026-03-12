# Annotation Refiner 项目说明

## 项目概述

Annotation Refiner 是一个用于检测框标注排查、修正和导出的桌面 Web 工具，支持交互式浏览问题样本、编辑框、清理异常标注，并将修正结果保存到输出目录。

当前主流程已支持两类标注格式：
- YOLO：`.txt`
- Pascal VOC：`.xml`

应用会优先按样本实际存在的标注文件格式读取；保存修正结果时，会尽量沿用 GT 原始格式输出到 `Output Path`。

## 安装说明

### 1. 环境要求
- Python 3.8+
- 依赖库见 [requirements.txt](C:/Users/admin/Desktop/dingding/refiner_20260305/refiner/requirements.txt)

### 2. 安装步骤
```bash
# 克隆项目
git clone https://github.com/guenqing/LableOpt.git

# 进入项目目录
cd LableOpt

# 安装依赖
python -m pip install -r requirements.txt

# 启动应用
python anno_refiner_app/main.py
```

### 3. 访问应用
启动后，在浏览器中访问显示的地址（默认 `http://localhost:8080`）。

## 功能特性

### 核心能力
1. 交互式查看 GT / Pred 标注框
2. 编辑、移动、缩放、删除和新建标注框
3. Auto Focus 自动聚焦当前图片中的框
4. Clean Annotations 自动清理重复框、大框套小框和相似框
5. Dashboard 统计待处理样本、输入输出目录状态和问题样本
6. Cleanlab 问题样本分析与可视化

### 当前支持的标注格式

| 场景 | YOLO `.txt` | Pascal VOC `.xml` |
|------|-------------|-------------------|
| Dashboard 统计 | 支持 | 支持 |
| 标注页加载 GT | 支持 | 支持 |
| 标注页加载 Pred | 支持 | 支持 |
| Run Analysis | 支持 | 支持 |
| Output Path 临时保存 / 最终保存 | 支持 | 支持 |
| 独立脚本 `process_annotations.py` | 支持 | 支持 |

## 标注格式说明

### 1. YOLO `.txt`
每行格式：
```text
class_id x_center y_center width height
```
预测框可额外带置信度：
```text
class_id x_center y_center width height confidence
```

### 2. Pascal VOC `.xml`
使用常见的 `annotation/object/bndbox` 结构。

XML 中的类别名处理规则：
- 如果 `<name>` 本身就是数字字符串，例如 `0`、`1`，可直接使用。
- 如果 `<name>` 是字符串类别名，例如 `cat`、`dog`，建议在首页提供 `Classes File`，用于建立类名与类 ID 的映射。
- 没有映射文件时，字符串类名 XML 可能无法参与分析，但仍可影响部分读取行为。

## 快捷键说明

### 全局快捷键
| 快捷键 | 功能 | 说明 |
|--------|------|------|
| `Ctrl + Z` | 撤销 | 撤销上一步框编辑操作 |
| `Ctrl + Y` | 重做 | 重做上一步撤销 |
| `Ctrl + Shift + Z` | 重做 | 与 `Ctrl + Y` 等价 |
| `=` 或 `+` | 放大 | 放大当前视图 |
| `-` | 缩小 | 缩小当前视图 |
| `0` | 重置缩放 | 恢复到 1x 视图 |
| `[` | 上一张 | 切换到上一张样本 |
| `]` | 下一张 | 切换到下一张样本 |

### 标注框选择与编辑
| 快捷键 | 功能 | 说明 |
|--------|------|------|
| `·` / `` ` `` | 循环选框 | 在当前可编辑框之间循环切换 |
| `Delete` / `Backspace` | 删除选中框 | 仅删除当前可编辑框 |
| `1 / 2 / 3 / 4` | 设置类别 | 将选中框类别改为 `0 / 1 / 2 / 3`；若当前无选框，则修改新建框默认类别 |
| `方向键` | 微移选中框 | 每次移动 1 像素 |
| `Shift + 方向键` | 快速移动选中框 | 每次移动 10 像素 |
| `a / s / d / f` | 调整单边 | 分别调整左 / 上 / 右 / 下边界，默认向外扩 1 像素 |
| `A / S / D / F` | 反向调整单边 | 对应边界向内收 1 像素 |
| `z / x / c / v` | 调整四角 | 分别调整左上 / 右上 / 右下 / 左下角，默认向外扩 1 像素 |
| `Z / X / C / V` | 反向调整四角 | 对应角点向内收 1 像素 |

### 显示与编辑状态切换
| 快捷键 | 功能 | 说明 |
|--------|------|------|
| `Q` | 显示/隐藏 GT | 切换 GT 框可见性 |
| `W` | 显示/隐藏 Pred | 切换 Pred 框可见性 |
| `E` | 交换可编辑 | 将当前可编辑框与参考框互换编辑状态 |
| `R` | 清空可编辑 | 删除所有当前可编辑框 |
| `T` | 激活参考 | 将当前参考框批量切换为可编辑 |

### 跨帧辅助快捷键
| 快捷键 | 功能 | 说明 |
|--------|------|------|
| `Y` | Toggle Extend GT to Next | 开关“将当前可编辑 GT 传递到下一帧” |
| `U` | Toggle Prefer Previous | 开关“重叠时优先保留上一帧结果”；仅在 Extend GT to Next 开启时有效 |

## 使用指南

### 1. 首页配置
在首页填写以下路径：
- `Images Path`：图片根目录
- `GT Labels Path`：GT 标注根目录
- `Pred Labels Path`：预测标注根目录，可选
- `Output Path`：修正结果输出目录，必填
- `Human Verified Annotation Path`：人工确认结果目录，可选
- `Classes File`：类别映射文件，可选，推荐在 XML 使用字符串类名时提供

### 2. 目录与命名约定
图片与标注应保持相同的相对目录结构和文件主名，例如：
```text
images/a/b/frame_0001.jpg
gt/a/b/frame_0001.txt
pred/a/b/frame_0001.txt
```
或：
```text
images/a/b/frame_0001.jpg
gt/a/b/frame_0001.xml
pred/a/b/frame_0001.xml
```

也支持 GT 和 Pred 混合格式共存，只要相对路径主名一致即可，例如 GT 为 `.xml`、Pred 为 `.txt`。

### 3. Classes File 的建议格式
推荐两种：
- `classes.txt`
- `data.yaml`

`classes.txt` 示例：
```text
cat
dog
bird
```

`data.yaml` 示例：
```yaml
names:
  0: cat
  1: dog
  2: bird
```

### 4. Parse Data / Run Analysis
点击 `Parse Data` 后，系统会统计：
- 图片数量
- GT / Pred 可用样本数量
- Output / Human Verified 已处理数量
- Pending Samples

点击 `RUN ANALYSIS` 后，系统会：
1. 找到同时满足“图片存在、GT 存在、Pred 存在”的样本
2. 排除已存在于 `Output Path` 或 `Human Verified` 的样本
3. 将 GT / Pred 转为 Cleanlab 所需格式
4. 生成待处理问题样本列表

### 5. 进入标注页
点击 `Start Annotation` 后可进入标注页面：
- GT 默认可编辑
- Pred 默认作为参考框显示
- 可使用按钮或快捷键切换样本
- Auto Focus 会自动缩放到当前图片中的框区域

### 6. 标注框操作
当前页面中的框操作规则如下：
- 左键点击框：选中该框；若多个可编辑框重叠，优先选中更小的框
- 左键拖动已选中框：移动框位置
- 左键拖动控制点：缩放框尺寸
- 在空白区域按住左键拖动：创建新框
- 新建框默认写入当前默认类别；如果没有选中框，可先按 `1/2/3/4` 修改默认类别
- 可编辑框为实线，参考框通常为虚线
- 删除、改类、微调边和角点操作都只对当前可编辑框生效
- 当视图放大后，可通过导航区和滚动条辅助定位

### 7. 保存行为
点击 `Save` 后，系统会将当前可编辑框保存到 `Output Path`：
- 如果 GT 原始格式是 `.txt`，输出为 `*_tmp.txt`
- 如果 GT 原始格式是 `.xml`，输出为 `*_tmp.xml`

导航到上一张/下一张时：
- 若启用了 Auto Save，会自动保存当前结果
- 若启用了 Save Unmodified，未改动样本也会按当前状态保存

确认保留修改后：
- `*_tmp.txt` 会变成 `.txt`
- `*_tmp.xml` 会变成 `.xml`

## XML 使用建议

### 推荐场景
如果你的 XML 标注满足以下条件，建议直接使用当前主流程：
- Pascal VOC 常见结构
- 每个对象使用 `<object>` + `<name>` + `<bndbox>`
- 坐标字段为 `xmin/ymin/xmax/ymax`
- 类别名能通过 `Classes File` 映射到稳定 class id

### 建议提供 Classes File 的情况
以下情况建议一定提供 `Classes File`：
- XML 的 `<name>` 为自然语言类名，例如 `person`、`car`
- 你需要运行 `RUN ANALYSIS`
- 你希望 XML 与 YOLO 数据混合使用时类别保持一致

### 当前限制
- 新建框和修改类别时，UI 内部仍使用数字 `class_id` 进行编辑。
- 因此 XML 若使用字符串类名，最好配套 `Classes File`，否则分析阶段可能无法推断稳定类别编号。
- 非标准 XML 结构目前不保证支持。

## 高级功能

### Clean Annotations
自动清理以下类型的标注框：
1. 重复标注
2. 大框套小框
3. 不同类别但位置高度相似的框

### Auto Focus
- 对有框图片自动缩放到主要框区域
- 对窄图 / 竖图，缩放后会在未铺满的轴上自动居中显示

### Box List 面板
右侧显示当前图片所有框：
- 来源：GT / Pred
- 类别
- 尺寸
- 选中状态

## 项目结构

```text
LableOpt/
├── anno_refiner_app/
│   ├── src/
│   │   ├── core/
│   │   │   ├── analyzer.py
│   │   │   ├── file_manager.py
│   │   │   ├── label_utils.py
│   │   │   └── yolo_utils.py
│   │   ├── ui/
│   │   └── ...
│   └── main.py
├── process_annotations.py
├── CODE_CHANGE_LOG.md
├── README.md
└── requirements.txt
```

## 常见问题

### 1. Pending Samples 为 0
可能原因：
- 图片和标注相对路径对不上
- GT / Pred 没有同名样本
- 样本已经在 `Output Path` 或 `Human Verified` 中存在
- XML / TXT 虽然存在，但目录结构不一致

### 2. XML 能显示，但分析报类别错误
可能原因：
- XML 的 `<name>` 是字符串类名
- 没有提供 `Classes File`
- `Classes File` 中的类别顺序 / 编号与 XML 类名不一致

建议：
- 配置 `classes.txt` 或 `data.yaml`
- 确保 XML 中的类名都能映射到唯一 class id

### 3. Output Path 里为什么有 `_tmp.xml` 或 `_tmp.txt`
这是暂存结果，用于区分“尚未最终确认”的修改。确认保留后会转换成正式文件。

### 4. 目前是不是所有 XML 都支持
不是。当前主要支持标准 Pascal VOC 风格 XML。若 XML 结构自定义较多，建议先转换或扩展解析逻辑。

## 更新记录

详细代码变更记录见 [CODE_CHANGE_LOG.md](C:/Users/admin/Desktop/dingding/refiner_20260305/refiner/CODE_CHANGE_LOG.md)。

## 许可证

MIT License

---

**作者**：guenqing  
**邮箱**：guenqing1007@gmail.com  
**最后更新**：2026-03-12
