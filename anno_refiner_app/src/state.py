"""
Global state management for the Refiner application.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from .models import IssueItem, IssueType, SessionConfig, ClassMapping


@dataclass
class AnalysisResults:
    """Analysis results from Cleanlab"""
    overlooked: List[IssueItem] = field(default_factory=list)
    swapped: List[IssueItem] = field(default_factory=list)
    bad_located: List[IssueItem] = field(default_factory=list)
    
    def get_by_type(self, issue_type: IssueType) -> List[IssueItem]:
        """Get issues by type"""
        mapping = {
            IssueType.OVERLOOKED: self.overlooked,
            IssueType.SWAPPED: self.swapped,
            IssueType.BAD_LOCATED: self.bad_located,
        }
        return mapping.get(issue_type, [])
    
    def set_by_type(self, issue_type: IssueType, items: List[IssueItem]):
        """Set issues by type"""
        if issue_type == IssueType.OVERLOOKED:
            self.overlooked = items
        elif issue_type == IssueType.SWAPPED:
            self.swapped = items
        elif issue_type == IssueType.BAD_LOCATED:
            self.bad_located = items


@dataclass
class AppState:
    """Application state"""
    # Configuration (backup disabled by default)
    config: SessionConfig = field(default_factory=lambda: SessionConfig(backup_enabled=False))
    
    # Analysis results
    results: AnalysisResults = field(default_factory=AnalysisResults)
    
    # Analysis status
    is_analyzing: bool = False
    analysis_progress: float = 0.0
    analysis_message: str = ""
    analysis_complete: bool = False
    
    # Path validation results
    path_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Class mapping (loaded from classes file)
    class_mapping: Optional[ClassMapping] = None
    
    # Selected issue types for annotation
    selected_overlooked: bool = True
    selected_swapped: bool = True
    selected_bad_located: bool = True
    
    # Annotation queue (merged from selected issue types)
    annotation_queue: List[IssueItem] = field(default_factory=list)
    current_annotation_index: int = 0
    
    def get_selected_issues(self) -> List[IssueItem]:
        """Get merged list of selected issue types (deduplicated by image_path)"""
        seen_paths = set()
        merged = []
        
        if self.selected_overlooked:
            for item in self.results.overlooked:
                if item.image_path not in seen_paths:
                    seen_paths.add(item.image_path)
                    merged.append(item)
        
        if self.selected_swapped:
            for item in self.results.swapped:
                if item.image_path not in seen_paths:
                    seen_paths.add(item.image_path)
                    merged.append(item)
        
        if self.selected_bad_located:
            for item in self.results.bad_located:
                if item.image_path not in seen_paths:
                    seen_paths.add(item.image_path)
                    merged.append(item)
        
        return merged
    
    def reset_analysis(self):
        """Reset analysis state"""
        self.results = AnalysisResults()
        self.is_analyzing = False
        self.analysis_progress = 0.0
        self.analysis_message = ""
        self.analysis_complete = False
        self.annotation_queue = []
        self.current_annotation_index = 0


# Global application state instance
app_state = AppState()
