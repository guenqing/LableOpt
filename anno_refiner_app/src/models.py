from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
from enum import Enum
import uuid


class IssueType(Enum):
    OVERLOOKED = "overlooked"
    SWAPPED = "swapped"
    BAD_LOCATED = "bad_located"


class BoxSource(Enum):
    GT = "gt"
    PRED = "pred"


@dataclass
class BBox:
    """UI layer bounding box"""
    x: float                    # top-left x (pixel)
    y: float                    # top-left y (pixel)
    w: float                    # width (pixel)
    h: float                    # height (pixel)
    class_id: int               # class ID
    source: BoxSource           # source
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    selected: bool = False
    visible: bool = True        # whether to render this box
    editable: bool = True       # whether this box can be edited


@dataclass
class IssueItem:
    """Issue sample"""
    image_path: str             # relative path: category/video/frame_xxxxx.jpg
    issue_type: IssueType       # issue type
    score: float                # issue severity score (lower = more severe)
    box_index: Optional[int]    # issue box index (for highlighting)


@dataclass  
class SessionConfig:
    """Session configuration"""
    images_path: str = ""
    pred_labels_path: str = ""
    gt_labels_path: str = ""
    classes_file: str = ""      # optional: classes.txt or data.yaml path
    top_k: int = 10
    backup_enabled: bool = False  # Default to False


@dataclass
class ClassMapping:
    """Class mapping"""
    id_to_name: Dict[int, str]

    @classmethod
    def from_file(cls, file_path: str) -> 'ClassMapping':
        """Load class mapping from classes.txt or data.yaml"""
        path = Path(file_path)
        id_to_name = {}

        if path.suffix == '.txt':
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        id_to_name[i] = name
        elif path.suffix in ('.yaml', '.yml'):
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            names = data.get('names', {})
            if isinstance(names, list):
                id_to_name = {i: n for i, n in enumerate(names)}
            elif isinstance(names, dict):
                id_to_name = {int(k): v for k, v in names.items()}

        return cls(id_to_name=id_to_name)

    def get_name(self, class_id: int) -> str:
        """Get class name, return str(class_id) if not found"""
        return self.id_to_name.get(class_id, str(class_id))

    def get_display(self, class_id: int) -> str:
        """Get display string: 'id: name' or 'id'"""
        if class_id in self.id_to_name:
            return f"{class_id}: {self.id_to_name[class_id]}"
        return str(class_id)
