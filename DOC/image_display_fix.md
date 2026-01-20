# 横向与纵向图片显示问题修复文档

> 创建日期: 2026-01-16
> 状态: 已修复

## 一、问题描述

在标注工具（Annotator）中，Viewer和Navigator组件在处理不同宽高比的图像时存在显示问题：

### 问题A：高>宽的图像（纵向图像）
- **Viewer显示**：✅ 理想 - 图像完整显示，填充高度，水平居中
- **Navigator显示**：❌ 问题 - 图像非常小，没有填充效果

### 问题B：宽>高的图像（横向图像）
- **Viewer显示**：❌ 问题 - 图像被压缩到左上角，没有填充效果
- **Navigator显示**：❌ 问题 - 图像非常小，没有填充效果

## 二、根本原因

1. **图像元素缺少明确尺寸**：`interactive_image` 组件需要明确的宽度和高度，transform才能正确工作。
2. **1x缩放时未应用transform**：在 `_apply_transform()` 中，当 `zoom <= 1.0` 时设置了 `transform: none;`，导致图像没有被缩放和居中。
3. **Navigator未应用transform**：`_update_minimap()` 方法没有应用transform来缩放和居中图像。

## 三、修复方案

### 3.1 设置图像元素的明确尺寸

**位置**: `src/ui/components.py` 的 `load_image()` 方法

**修复内容**:
```python
if self.image_component:
    self.image_component.set_source(image_path)
    # Set explicit dimensions for the image element so transform works correctly
    self.image_component.style(f'width: {self.image_width}px; height: {self.image_height}px; display: block;')

if hasattr(self, 'minimap_component') and self.minimap_component:
    self.minimap_component.set_source(image_path)
    # Set explicit dimensions for the minimap image element
    self.minimap_component.style(f'width: {self.image_width}px; height: {self.image_height}px; display: block;')
```

**原理**: `interactive_image` 组件需要明确的尺寸才能让CSS transform正确工作。通过设置原始图像尺寸，transform可以基于这些尺寸进行缩放和居中计算。

### 3.2 修复Viewer在1x缩放时的transform

**位置**: `src/ui/components.py` 的 `_apply_transform()` 方法

**修复前**:
```python
if self.zoom <= 1.0:
    # At 1x zoom, no transform needed
    self.transform_container.style('transform: none;')
```

**修复后**:
```python
if self.zoom <= 1.0:
    # At 1x zoom, scale and center the image
    transform = f'translate({display_x_1x}px, {display_y_1x}px) scale({scale_1x})'
    self.transform_container.style(f'transform: {transform}; transform-origin: 0 0;')
```

**原理**: 即使在1x缩放时，也需要通过transform来缩放和居中图像。`scale_1x` 是使图像适应容器的缩放比例，`display_x_1x` 和 `display_y_1x` 是居中偏移量。

### 3.3 修复Navigator的transform

**位置**: `src/ui/components.py` 的 `_update_minimap()` 方法

**修复前**:
```python
def _update_minimap(self) -> None:
    """Update minimap (no viewport indicator - removed for simplicity)"""
    if not hasattr(self, 'minimap_component') or not self.minimap_component:
        return
    # No overlay needed - Navigator is used only for click-to-navigate
    self.minimap_component.set_content('')
```

**修复后**:
```python
def _update_minimap(self) -> None:
    """Update minimap transform to scale and center the image"""
    if not hasattr(self, 'minimap_component') or not self.minimap_component:
        return
    if not hasattr(self, 'minimap_transform_container') or not self.minimap_transform_container:
        return
    if self.image_width <= 0 or self.image_height <= 0:
        return
    
    # Calculate scale to fit image in minimap container while maintaining aspect ratio
    scale_x = self.minimap_width / self.image_width
    scale_y = self.minimap_height / self.image_height
    scale = min(scale_x, scale_y)  # Use smaller scale to fit both dimensions
    
    # Calculate actual displayed size
    display_width = self.image_width * scale
    display_height = self.image_height * scale
    
    # Center the image in the minimap container
    display_x = (self.minimap_width - display_width) / 2
    display_y = (self.minimap_height - display_height) / 2
    
    # Apply transform to scale and center the image
    transform = f'translate({display_x}px, {display_y}px) scale({scale})'
    self.minimap_transform_container.style(f'transform: {transform}; transform-origin: 0 0;')
    
    # No overlay needed - Navigator is used only for click-to-navigate
    self.minimap_component.set_content('')
```

**原理**: Navigator使用与Viewer相同的逻辑来计算缩放比例和居中偏移，然后通过transform应用到图像上。

### 3.4 修复zoom>1时的缩放逻辑

**位置**: `src/ui/components.py` 的 `_apply_transform()` 方法

**修复内容**: 在zoom>1时，使用 `scale({self.zoom * scale_1x})` 来正确缩放图像，因为图像元素是原始尺寸，需要先应用scale_1x，然后再应用zoom。

### 3.5 修复create_navigator中的图像加载

**位置**: `src/ui/components.py` 的 `create_navigator()` 方法

**修复内容**: 在创建Navigator时，如果图像已经加载，也设置图像尺寸并更新minimap。

## 四、技术细节

### 4.1 Transform计算逻辑

**Viewer (1x缩放)**:
1. 计算 `scale_1x = min(view_width/image_width, view_height/image_height)`
2. 计算显示尺寸: `display_width = image_width * scale_1x`, `display_height = image_height * scale_1x`
3. 计算居中偏移: `display_x = (view_width - display_width) / 2`, `display_y = (view_height - display_height) / 2`
4. 应用transform: `translate(display_x, display_y) scale(scale_1x)`

**Viewer (zoom>1)**:
1. 计算 `scale_1x` 和居中偏移（同上）
2. 计算平移量: `translate_x = display_x_1x - pan_x * scale_1x * zoom`
3. 应用transform: `translate(translate_x, translate_y) scale(zoom * scale_1x)`

**Navigator**:
1. 计算 `scale = min(minimap_width/image_width, minimap_height/image_height)`
2. 计算显示尺寸和居中偏移（同上）
3. 应用transform: `translate(display_x, display_y) scale(scale)`

### 4.2 CSS Transform的工作原理

CSS transform按照从右到左的顺序应用：
- `transform: translate(x, y) scale(z)` 意味着：先缩放，然后平移
- `transform-origin: 0 0` 设置变换的原点为左上角

## 五、测试验证

### 5.1 测试用例

1. **横向图片（1280x640）**：
   - ✅ Viewer填充宽度，垂直居中
   - ✅ Navigator尽可能填充，居中显示

2. **纵向图片（640x1280）**：
   - ✅ Viewer填充高度，水平居中
   - ✅ Navigator尽可能填充，居中显示

3. **正方形图片（800x800）**：
   - ✅ Viewer和Navigator都正确显示

4. **缩放功能**：
   - ✅ 1x缩放时，图像正确显示和居中
   - ✅ zoom>1时，图像正确缩放和平移

### 5.2 验证方法

1. 启动应用并加载不同宽高比的图片
2. 检查Viewer中的图片是否完整显示、填充容器、居中
3. 检查Navigator中的图片是否完整显示、填充容器、居中
4. 测试缩放功能，确保在不同缩放级别下图片显示正确

## 六、相关文件

- `src/ui/components.py` - InteractiveAnnotator组件
  - `load_image()` - 图像加载和尺寸设置
  - `_apply_transform()` - Viewer的transform应用
  - `_update_minimap()` - Navigator的transform应用
  - `create_navigator()` - Navigator创建

## 七、参考资源

- [NiceGUI interactive_image 文档](https://nicegui.io/documentation/image)
- [CSS Transform 文档](https://developer.mozilla.org/en-US/docs/Web/CSS/transform)
- [object-fit 文档](https://developer.mozilla.org/en-US/docs/Web/CSS/object-fit)
