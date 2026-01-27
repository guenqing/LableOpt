#!/usr/bin/env python3
"""
Code Reviewer for Annotation Refiner

This tool analyzes the codebase to identify logical issues, potential bugs, and
to check if all aspects of the problem have been considered.
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

class CodeReviewer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = []
        self.reviewed_files = []
        
    def add_issue(self, file_path: str, line: int, severity: str, issue: str, suggestion: str = ""):
        """Add a review issue"""
        self.issues.append({
            "file": file_path,
            "line": line,
            "severity": severity,
            "issue": issue,
            "suggestion": suggestion
        })
    
    def review_file(self, file_path: Path):
        """Review a single file"""
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.reviewed_files.append(str(file_path.relative_to(self.project_root)))
            print(f"Reviewing: {file_path.relative_to(self.project_root)}")
            
            # Parse the file
            tree = ast.parse(content)
            lines = content.split('\n')
            
            # Extract file extension
            ext = file_path.suffix.lower()
            
            if ext == '.py':
                self.review_python_file(file_path, tree, lines, content)
            
        except Exception as e:
            print(f"Error reviewing {file_path}: {e}")
    
    def review_python_file(self, file_path: Path, tree: ast.AST, lines: List[str], content: str):
        """Review Python files"""
        rel_path = str(file_path.relative_to(self.project_root))
        
        # Check for specific modules
        if "core/yolo_utils.py" in rel_path:
            self.review_yolo_utils(file_path, tree, lines)
        elif "ui/components.py" in rel_path:
            self.review_components(file_path, tree, lines)
    
    def review_yolo_utils(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """Review yolo_utils.py specifically"""
        rel_path = str(file_path.relative_to(self.project_root))
        
        # Find and review yolo_to_pixel function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'yolo_to_pixel':
                self.review_yolo_to_pixel(rel_path, node, lines)
                
            elif isinstance(node, ast.FunctionDef) and node.name == 'pixel_to_yolo':
                self.review_pixel_to_yolo(rel_path, node, lines)
                
            elif isinstance(node, ast.FunctionDef) and node.name == 'get_image_size':
                self.review_get_image_size(rel_path, node, lines)
    
    def review_yolo_to_pixel(self, file_path: str, node: ast.FunctionDef, lines: List[str]):
        """Review yolo_to_pixel function"""
        # Check for proper validation
        has_img_dim_check = False
        has_coord_clamping = False
        has_min_size_check = False
        has_boundary_check = False
        
        for body_node in node.body:
            if isinstance(body_node, ast.If):
                # Check image dimension validation
                if self._contains_text(body_node, 'img_w <= 0 or img_h <= 0'):
                    has_img_dim_check = True
                
                # Check box dimension validation
                if self._contains_text(body_node, 'x1 >= x2') or self._contains_text(body_node, 'y1 >= y2'):
                    has_boundary_check = True
            
            # Check coordinate clamping
            if isinstance(body_node, ast.Assign):
                if self._contains_text(body_node, 'max\(0.0, min\('):
                    has_coord_clamping = True
                
                # Check minimum size validation
                if self._contains_text(body_node, 'max\(0.001'):
                    has_min_size_check = True
        
        if not has_img_dim_check:
            self.add_issue(file_path, node.lineno, "ERROR", 
                          "Missing image dimension validation", 
                          "Add check for img_w <= 0 or img_h <= 0")
        
        if not has_coord_clamping:
            self.add_issue(file_path, node.lineno, "ERROR", 
                          "Missing coordinate clamping to [0,1] range", 
                          "Add max(0.0, min(value, 1.0)) for all normalized coordinates")
        
        if not has_min_size_check:
            self.add_issue(file_path, node.lineno, "WARNING", 
                          "Missing minimum size validation", 
                          "Add check to ensure width and height are at least 0.001")
        
        if not has_boundary_check:
            self.add_issue(file_path, node.lineno, "ERROR", 
                          "Missing boundary check for box dimensions", 
                          "Add check to ensure x1 < x2 and y1 < y2")
    
    def review_pixel_to_yolo(self, file_path: str, node: ast.FunctionDef, lines: List[str]):
        """Review pixel_to_yolo function"""
        # Check for proper validation
        has_img_dim_check = False
        has_coord_clamping = False
        
        for body_node in node.body:
            if isinstance(body_node, ast.If):
                # Check image dimension validation
                if self._contains_text(body_node, 'img_w <= 0 or img_h <= 0'):
                    has_img_dim_check = True
            
            # Check coordinate clamping
            if isinstance(body_node, ast.Assign):
                if self._contains_text(body_node, 'max\(0.0, min\('):
                    has_coord_clamping = True
        
        if not has_img_dim_check:
            self.add_issue(file_path, node.lineno, "ERROR", 
                          "Missing image dimension validation", 
                          "Add check for img_w <= 0 or img_h <= 0")
        
        if not has_coord_clamping:
            self.add_issue(file_path, node.lineno, "ERROR", 
                          "Missing coordinate clamping to image boundaries", 
                          "Add max(0.0, min(value, img_dim)) for all pixel coordinates")
    
    def review_get_image_size(self, file_path: str, node: ast.FunctionDef, lines: List[str]):
        """Review get_image_size function"""
        # Check supported formats
        supported_formats = {'jpeg', 'png', 'bmp'}
        found_formats = set()
        
        for body_node in node.body:
            if isinstance(body_node, ast.If) and hasattr(body_node.test, 'left'):
                if hasattr(body_node.test.left, 'attr') and body_node.test.left.attr == 'startswith':
                    if self._contains_text(body_node.test, '\xff\xd8\xff'):
                        found_formats.add('jpeg')
                    elif self._contains_text(body_node.test, '\x89PNG'):
                        found_formats.add('png')
                    elif self._contains_text(body_node.test, 'BM'):
                        found_formats.add('bmp')
        
        missing_formats = supported_formats - found_formats
        if missing_formats:
            self.add_issue(file_path, node.lineno, "WARNING", 
                          f"Missing support for formats: {', '.join(missing_formats)}", 
                          "Consider adding support for additional image formats")
    
    def review_components(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """Review components.py specifically"""
        rel_path = str(file_path.relative_to(self.project_root))
        
        # Find and review specific methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == 'auto_focus_boxes':
                    self.review_auto_focus_boxes(rel_path, node, lines)
                elif node.name == '_constrain_pan':
                    self.review_constrain_pan(rel_path, node, lines)
    
    def review_auto_focus_boxes(self, file_path: str, node: ast.FunctionDef, lines: List[str]):
        """Review auto_focus_boxes method"""
        # Check if all boxes are included
        has_all_boxes = False
        has_symmetric_buffer = False
        
        for body_node in node.body:
            if isinstance(body_node, ast.Assign) and hasattr(body_node, 'value'):
                if self._contains_text(body_node.value, 'self.gt_boxes + self.pred_boxes'):
                    has_all_boxes = True
                
                # Check for symmetric buffer
                if hasattr(body_node.value, 'id') and body_node.value.id == 'buffer':
                    has_symmetric_buffer = True
        
        if not has_all_boxes:
            self.add_issue(file_path, node.lineno, "ERROR", 
                          "Not all boxes are included in boundary calculation", 
                          "Use self.gt_boxes + self.pred_boxes to include all boxes")
        
        if not has_symmetric_buffer:
            self.add_issue(file_path, node.lineno, "WARNING", 
                          "潜在的不对称缓冲区计算", 
                          "确保缓冲区在所有方向上均匀应用")
    
    def review_constrain_pan(self, file_path: str, node: ast.FunctionDef, lines: List[str]):
        """Review _constrain_pan method"""
        # Check for overscroll allowance
        has_overscroll = False
        
        for body_node in node.body:
            if isinstance(body_node, ast.Assign):
                if self._contains_text(body_node, 'overscroll') or self._contains_text(body_node, '0.1'):
                    has_overscroll = True
        
        if not has_overscroll:
            self.add_issue(file_path, node.lineno, "WARNING", 
                          "平移时没有滚动余量", 
                          "添加滚动余量以提高边缘框的可见性")
    
    def _contains_text(self, node: ast.AST, text: str) -> bool:
        """Check if a node contains specific text"""
        return text in ast.dump(node)
    
    def run(self):
        """Run the code review on the entire project"""
        print(f"正在为项目开始代码审查: {self.project_root}")
        print("=" * 60)
        
        # Review Python files
        for file_path in self.project_root.rglob("*.py"):
            if "__pycache__" not in str(file_path) and "venv" not in str(file_path):
                self.review_file(file_path)
        
        print("\n" + "=" * 60)
        print("代码审查总结")
        print("=" * 60)
        
        if not self.issues:
            print("未发现任何问题")
        else:
            print(f"发现 {len(self.issues)} 个问题：")
            print()
            
            # Group issues by file
            issues_by_file = {}
            for issue in self.issues:
                file = issue["file"]
                if file not in issues_by_file:
                    issues_by_file[file] = []
                issues_by_file[file].append(issue)
            
            for file, file_issues in issues_by_file.items():
                print(f"{file}:")
                for issue in file_issues:
                    print(f"  [{issue['severity']}] Line {issue['line']}: {issue['issue']}")
                    if issue['suggestion']:
                        print(f"    建议：{issue['suggestion']}")
                print()
        
        print(f"已审查 {len(self.reviewed_files)} 个文件")

if __name__ == "__main__":
    # Run the code reviewer on the current project
    project_root = Path(__file__).parent
    reviewer = CodeReviewer(project_root)
    reviewer.run()
