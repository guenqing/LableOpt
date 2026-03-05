#!/usr/bin/env python3
"""
处理标注文件中的问题：
1. 重复标注：删除小的标注框
2. 大框框住多个小框：删除大框
3. 不同类别标注框位置相似：删除两个标注框

支持两种格式：
- YOLO格式：class_id x_center y_center width height
- XML格式：PASCAL VOC格式
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """计算两个边界框的IoU
    
    Args:
        box1: (xmin, ymin, xmax, ymax)
        box2: (xmin, ymin, xmax, ymax)
    
    Returns:
        IoU值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_area(box: Tuple[int, int, int, int]) -> int:
    """计算边界框的面积
    
    Args:
        box: (xmin, ymin, xmax, ymax)
    
    Returns:
        面积
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def is_box_inside(inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int]) -> bool:
    """判断一个边界框是否完全在另一个边界框内
    
    Args:
        inner_box: 内部边界框 (xmin, ymin, xmax, ymax)
        outer_box: 外部边界框 (xmin, ymin, xmax, ymax)
    
    Returns:
        是否在内部
    """
    return (inner_box[0] >= outer_box[0] and
            inner_box[1] >= outer_box[1] and
            inner_box[2] <= outer_box[2] and
            inner_box[3] <= outer_box[3])


def calculate_boundary_diffs(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> List[int]:
    """计算两个边界框的边界差值
    
    Args:
        box1: (xmin, ymin, xmax, ymax)
        box2: (xmin, ymin, xmax, ymax)
    
    Returns:
        [xmin_diff, ymin_diff, xmax_diff, ymax_diff]
    """
    return [
        abs(box1[0] - box2[0]),  # xmin差异
        abs(box1[1] - box2[1]),  # ymin差异
        abs(box1[2] - box2[2]),  # xmax差异
        abs(box1[3] - box2[3])   # ymax差异
    ]


def is_boundary_similar(diffs: List[int], threshold: int = 10) -> bool:
    """判断边界是否相似（任意三边满足阈值）
    
    Args:
        diffs: 边界差值列表 [xmin_diff, ymin_diff, xmax_diff, ymax_diff]
        threshold: 阈值
    
    Returns:
        是否相似
    """
    # 检查任意三边是否小于等于阈值
    sorted_diffs = sorted(diffs)
    return sum(1 for d in sorted_diffs[:3] if d <= threshold) >= 3


class AnnotationProcessor:
    """标注文件处理器"""
    
    def __init__(self, label_dir: str):
        self.label_dir = label_dir
        self.report = []
    
    def process(self):
        """处理所有标注文件"""
        for filename in os.listdir(self.label_dir):
            file_path = os.path.join(self.label_dir, filename)
            
            if filename.endswith('.txt'):
                self._process_yolo_file(file_path)
            elif filename.endswith('.xml'):
                self._process_xml_file(file_path)
        
        # 生成报告
        self._generate_report()
    
    def _process_yolo_file(self, file_path: str):
        """处理YOLO格式标注文件"""
        try:
            # 读取YOLO标注
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析YOLO标注为边界框
            boxes = []
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 转换为绝对坐标（假设图像大小为1x1，实际处理时需要根据图像大小调整）
                    xmin = x_center - width / 2
                    ymin = y_center - height / 2
                    xmax = x_center + width / 2
                    ymax = y_center + height / 2
                    
                    boxes.append({
                        'id': i,
                        'class_id': class_id,
                        'box': (xmin, ymin, xmax, ymax),
                        'area': width * height
                    })
            
            # 处理标注框
            filtered_boxes = self._process_boxes(boxes)
            
            # 保存处理后的标注
            with open(file_path, 'w', encoding='utf-8') as f:
                for box in filtered_boxes:
                    xmin, ymin, xmax, ymax = box['box']
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    width = xmax - xmin
                    height = ymax - ymin
                    f.write(f"{box['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
        except Exception as e:
            self.report.append(f"Error processing YOLO file {file_path}: {str(e)}")
    
    def _process_xml_file(self, file_path: str):
        """处理XML格式标注文件"""
        try:
            # 解析XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 获取图像大小
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # 解析标注框
            boxes = []
            objects = root.findall('object')
            for i, obj in enumerate(objects):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                boxes.append({
                    'id': i,
                    'class_id': name,
                    'box': (xmin, ymin, xmax, ymax),
                    'area': calculate_area((xmin, ymin, xmax, ymax)),
                    'object': obj
                })
            
            # 处理标注框
            filtered_boxes = self._process_boxes(boxes)
            
            # 移除所有object标签
            for obj in root.findall('object'):
                root.remove(obj)
            
            # 添加过滤后的object标签
            for box in filtered_boxes:
                root.append(box['object'])
            
            # 保存处理后的XML
            tree.write(file_path, encoding='utf-8', xml_declaration=True)
            
        except Exception as e:
            self.report.append(f"Error processing XML file {file_path}: {str(e)}")
    
    def _process_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """处理标注框，解决重复、大框框小框、相似标注的问题"""
        # 标记要删除的框
        to_delete = set()
        
        # 1. 处理重复标注（删除小的）
        for i in range(len(boxes)):
            if i in to_delete:
                continue
            for j in range(i + 1, len(boxes)):
                if j in to_delete:
                    continue
                
                box1 = boxes[i]['box']
                box2 = boxes[j]['box']
                
                # 计算IoU
                iou = calculate_iou(box1, box2)
                
                # 如果IoU大于0.7，认为是重复标注
                if iou > 0.7:
                    # 删除面积小的
                    if boxes[i]['area'] < boxes[j]['area']:
                        to_delete.add(i)
                        self.report.append(f"Deleted duplicate box {i} (smaller area)")
                    else:
                        to_delete.add(j)
                        self.report.append(f"Deleted duplicate box {j} (smaller area)")
        
        # 2. 处理大框框住多个小框的情况
        # 先过滤掉已标记删除的框
        remaining_boxes = [box for i, box in enumerate(boxes) if i not in to_delete]
        
        # 标记新的要删除的框
        new_to_delete = set()
        
        for i in range(len(remaining_boxes)):
            if i in new_to_delete:
                continue
            
            # 计算有多少小框在当前框内
            contained_count = 0
            for j in range(len(remaining_boxes)):
                if i == j or j in new_to_delete:
                    continue
                
                if is_box_inside(remaining_boxes[j]['box'], remaining_boxes[i]['box']):
                    contained_count += 1
            
            # 如果框住了2个或以上小框，删除大框
            if contained_count >= 2:
                new_to_delete.add(i)
                self.report.append(f"Deleted large box {i} (contains {contained_count} smaller boxes)")
        
        # 再次过滤
        remaining_boxes = [box for i, box in enumerate(remaining_boxes) if i not in new_to_delete]
        
        # 3. 处理不同类别标注框位置相似的情况
        final_to_delete = set()
        
        for i in range(len(remaining_boxes)):
            if i in final_to_delete:
                continue
            for j in range(i + 1, len(remaining_boxes)):
                if j in final_to_delete:
                    continue
                
                # 检查是否是不同类别
                if remaining_boxes[i]['class_id'] != remaining_boxes[j]['class_id']:
                    box1 = remaining_boxes[i]['box']
                    box2 = remaining_boxes[j]['box']
                    
                    # 计算IoU
                    iou = calculate_iou(box1, box2)
                    
                    # 计算边界差值
                    diffs = calculate_boundary_diffs(box1, box2)
                    
                    # 如果IoU大于0.7且边界相似，删除两个框
                    if iou > 0.7 and is_boundary_similar(diffs):
                        final_to_delete.add(i)
                        final_to_delete.add(j)
                        self.report.append(f"Deleted boxes {i} and {j} (different classes with similar positions)")
        
        # 最终过滤
        filtered_boxes = [box for i, box in enumerate(remaining_boxes) if i not in final_to_delete]
        
        return filtered_boxes
    
    def _generate_report(self):
        """生成处理报告"""
        report_path = os.path.join(self.label_dir, 'processing_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Annotation Processing Report\n")
            f.write("=" * 50 + "\n")
            for line in self.report:
                f.write(line + "\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total processing actions: {len(self.report)}")
        
        print(f"Processing report generated at: {report_path}")


if __name__ == "__main__":
    # 标注文件目录
    label_dir = "C:\\Users\\admin\\Desktop\\test1\\lable"
    
    # 创建处理器并执行
    processor = AnnotationProcessor(label_dir)
    processor.process()
    print("Annotation processing completed!")
