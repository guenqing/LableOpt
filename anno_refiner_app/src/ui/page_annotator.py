"""
Annotator page - Interactive annotation editing interface
"""
import asyncio
from pathlib import Path
from typing import List, Optional
import logging

from nicegui import ui

from ..state import app_state
from ..models import BBox, BoxSource, IssueItem, IssueType
from ..core.label_utils import (
    build_class_id_to_name,
    build_class_name_to_id,
    get_image_size,
    read_label_file,
    resolve_label_path,
)
from ..core.extend_gt_utils import (
    editable_boxes_to_yolo,
    apply_extend_gt_to_next,
    boxes_to_backup_payload,
    boxes_from_backup_payload,
    dumps_backup_payload,
    loads_backup_payload,
)
from ..core.file_manager import save_tmp_annotation, get_tmp_files, confirm_changes_for_tmp_files
from .components import InteractiveAnnotator

logger = logging.getLogger(__name__)


class AnnotatorPage:
    """Annotator page for editing bounding box annotations"""
    
    def __init__(self):
        # Auto save flag
        self.auto_save_enabled: bool = True
        self.save_unmodified_enabled: bool = True
        self.auto_focus_enabled: bool = True
        self.extend_gt_to_next_enabled: bool = False
        # Sub-toggle for extend: prefer previous-frame labels on overlap
        self.extend_prefer_previous_on_overlap_enabled: bool = False
        
        # Current GT boxes (from annotator component)
        self.current_gt_boxes: List[BBox] = []
        
        # Track if current image has unsaved changes
        self.boxes_modified: bool = False

        # Extend GT to Next state (cached from previous frame)
        self._pending_extend_gt_to_next: bool = False
        self._cached_editable_yolo: List[dict] = []
        # Backup file for undoing extend on current frame (written only when extend is applied)
        self._extend_backup_path_for_current: Optional[Path] = None
        
        # UI components
        self.annotator: Optional[InteractiveAnnotator] = None
        self.title_label = None
        self.progress_label = None
        self.issue_info_label = None
        self.prev_button = None
        self.next_button = None
        self.save_button = None
        self.zoom_label = None
        self.zoom_slider = None
        self.show_gt_checkbox = None
        self.show_pred_checkbox = None
        self.auto_focus_checkbox = None
        self.extend_gt_to_next_checkbox = None
        self.extend_prefer_previous_on_overlap_checkbox = None
        self.auto_save_checkbox = None
        self.save_unmodified_checkbox = None
        
        # Box list panel components
        self.box_list_container = None
        
        # Annotation cleaning properties
        self.clean_annotations_checkbox = None
        self.clean_annotations_count = None
        self.cleared_boxes = []  # Stores boxes that were removed during cleaning
        self.cleaning_applied = False  # Tracks if cleaning has been applied
        
        # View state for images without boxes
        self.previous_view_state = None  # Stores view state from previous image with boxes
        
        # Keyboard listener reference
        self.page_keyboard = None
    
    def create(self):
        """Create the annotator page"""
        # Check if queue is empty
        if not app_state.annotation_queue:
            self._show_empty_queue()
            return
        
        # Add custom styles
        ui.add_head_html('''
        <style>
            .annotator-container {
                display: flex;
                flex-direction: column;
                height: 100vh;
            }
            .control-panel {
                background: #f9fafb;
                border-left: 1px solid #e5e7eb;
            }
        </style>
        ''')
        
        with ui.column().classes('w-full min-h-screen bg-gray-50'):
            # Header bar
            self._create_header()
            
            # Main content area: Viewer | Navigator | Box List | Config
            with ui.row().classes('w-full p-4 gap-4 items-start flex-nowrap'):
                # Column 1: Viewer (main image area)
                self._create_viewer_area()
                
                # Column 2: Navigator (minimap)
                self._create_navigator_area()
                
                # Column 3: Box List panel
                self._create_box_list_panel()
                
                # Column 4: Config panel (no title)
                self._create_control_panel()
        
        # Set up page-level keyboard listener for navigation
        self.page_keyboard = ui.keyboard(on_key=self._handle_page_keys, ignore=['input', 'select', 'textarea'])
        
        # Load current image
        self._load_current_image()
    
    def _show_empty_queue(self):
        """Show empty queue message"""
        with ui.column().classes('w-full min-h-screen items-center justify-center bg-gray-50'):
            ui.icon('inbox', size='64px').classes('text-gray-300')
            ui.label('No samples in annotation queue').classes('text-xl text-gray-500 mt-4')
            ui.label('Please run analysis and select issues from the dashboard first.').classes('text-gray-400')
            ui.button('Back to Dashboard', on_click=lambda: ui.navigate.to('/')).classes('mt-6').tooltip('返回仪表盘')
    
    def _create_header(self):
        """Create header bar with title and navigation info"""
        with ui.row().classes('w-full bg-white shadow px-4 py-2 items-center gap-4'):
            # Title and info (no back button - must use "Go Back to Analysis" to ensure save confirmation)
            with ui.column().classes('flex-grow gap-0'):
                self.title_label = ui.label('Loading...').classes('text-lg font-semibold text-gray-800')
                with ui.row().classes('gap-2 items-center'):
                    self.progress_label = ui.label('').classes('text-sm text-gray-500')
                    ui.label('|').classes('text-gray-300')
                    self.issue_info_label = ui.label('').classes('text-sm')
    
    def _create_viewer_area(self):
        """Create the Viewer (main image area) with fixed size"""
        # Fixed width container for Viewer - prevents layout shift on zoom
        with ui.column().classes('flex-shrink-0 gap-2'):
            # Viewer container with fixed size
            viewer_container = ui.column().classes('bg-white rounded-lg shadow p-2')
            
            # Create annotator (Viewer only, Navigator created separately)
            self.annotator = InteractiveAnnotator(
                on_change=self._on_boxes_changed,
                on_zoom_change=self._on_zoom_changed,
                on_display_change=self._on_display_change_from_annotator
            )
            # Create Viewer without Navigator (navigator_container=None means no navigator here)
            self.annotator.create_ui(viewer_container, fixed_width=900, fixed_height=600)
            
            # Navigation buttons below Viewer
            with ui.row().classes('w-full justify-center gap-4 mt-2'):
                self.prev_button = ui.button(
                    'Prev', 
                    icon='chevron_left',
                    on_click=self._on_prev
                ).props('outline').tooltip('上一个样本')
                
                self.next_button = ui.button(
                    'Next',
                    icon='chevron_right',
                    on_click=self._on_next
                ).props('outline icon-right').tooltip('下一个样本')
    
    def _create_navigator_area(self):
        """Create the Navigator (minimap) area"""
        # Navigator container - fixed size proportional to Viewer
        with ui.column().classes('flex-shrink-0 gap-1'):
            navigator_container = ui.column().classes('bg-white rounded-lg shadow p-2')
            # Create Navigator in this container
            if self.annotator:
                self.annotator.create_navigator(navigator_container)
    
    def _create_box_list_panel(self):
        """Create box list panel showing all GT and Pred boxes"""
        # Box List panel - height matches Config panel
        with ui.card().classes('w-48 flex-shrink-0'):
            with ui.column().classes('w-full p-3 gap-2'):
                ui.label('Box List').classes('text-sm font-bold text-gray-700')
                
                # Scrollable container for box list - use flex-grow to fill available height
                with ui.scroll_area().classes('w-full').style('height: 580px;') as scroll_area:
                    self.box_list_container = ui.column().classes('w-full gap-1')
    
    def _update_box_list(self):
        """Update the box list panel with current boxes"""
        if not self.box_list_container or not self.annotator:
            return
        
        self.box_list_container.clear()
        
        all_boxes = self.annotator.get_all_boxes()
        gt_count = 0
        pred_count = 0
        
        with self.box_list_container:
            for box in all_boxes:
                source = box.source.value if isinstance(box.source, BoxSource) else box.source
                is_gt = source == 'gt'
                
                # Index for display
                if is_gt:
                    idx = gt_count
                    gt_count += 1
                    prefix = 'GT'
                    color_class = 'text-green-600'
                    bg_class = 'bg-green-50'
                else:
                    idx = pred_count
                    pred_count += 1
                    prefix = 'Pred'
                    color_class = 'text-blue-600'
                    bg_class = 'bg-blue-50'
                
                # Check if this box is selected
                is_selected = self.annotator.selected_box_id == box.id
                selected_class = 'ring-2 ring-red-500' if is_selected else ''
                
                # Check visibility and editable status
                is_visible = getattr(box, 'visible', True)
                is_editable = getattr(box, 'editable', True)
                
                # Box row container
                with ui.row().classes(f'w-full items-center gap-1 p-1 rounded cursor-pointer {bg_class} {selected_class}') \
                        .on('click', lambda e, b=box: self._on_box_list_click(b)):
                    # Index and class info
                    with ui.column().classes('flex-grow gap-0'):
                        label_text = f'{prefix} #{idx}'
                        ui.label(label_text).classes(f'text-xs font-medium {color_class}')
                        class_text = f'Class: {box.class_id}'
                        if not is_editable:
                            class_text += ' (ref)'
                        ui.label(class_text).classes('text-xs text-gray-500')
                        # Add size information
                        size_text = f'Size: {int(box.w)}×{int(box.h)}'
                        ui.label(size_text).classes('text-xs text-gray-500')
                    
                    # Eye icon for visibility toggle
                    eye_icon = 'visibility' if is_visible else 'visibility_off'
                    eye_color = 'text-gray-600' if is_visible else 'text-gray-300'
                    ui.button(
                        icon=eye_icon,
                        on_click=lambda e, b=box: self._on_toggle_box_visibility(b)
                    ).props('flat dense size=xs').classes(eye_color)
    
    def _on_box_list_click(self, box: BBox):
        """Handle click on box list item"""
        if not self.annotator:
            return
        
        # Check if editable - only editable boxes can be selected
        is_editable = getattr(box, 'editable', True)
        if is_editable:
            self.annotator.select_box_by_id(box.id)
        
        # Always update the list to show highlight
        self._update_box_list()
    
    def _on_toggle_box_visibility(self, box: BBox):
        """Toggle visibility of a box"""
        if not self.annotator:
            return
        
        current_visible = getattr(box, 'visible', True)
        self.annotator.set_box_visible(box.id, not current_visible)
        self._update_box_list()
    
    def _create_control_panel(self):
        """Create control panel (Config) on the right side - no title"""
        with ui.card().classes('w-56 flex-shrink-0'):
            with ui.column().classes('w-full p-3 gap-3'):
                # Display Options
                ui.label('Display Options').classes('text-sm font-bold text-gray-700')
                with ui.column().classes('gap-1'):
                    self.auto_focus_checkbox = ui.checkbox(
                        'Auto Focus',
                        value=True,
                        on_change=self._on_auto_focus_change,
                    ).classes('text-sm')
                    self.show_gt_checkbox = ui.checkbox(
                        'Show GT', 
                        value=True,
                        on_change=self._on_display_change
                    ).classes('text-sm')
                    self.show_pred_checkbox = ui.checkbox(
                        'Show Pred', 
                        value=True,
                        on_change=self._on_display_change
                    ).classes('text-sm')
                
                ui.separator()
                
                # Zoom Controls
                ui.label('Zoom Controls').classes('text-sm font-bold text-gray-700')
                with ui.column().classes('gap-2 w-full'):
                    with ui.row().classes('items-center gap-2'):
                        self.zoom_label = ui.label('1x').classes('text-sm font-mono w-8')
                        ui.button(icon='remove', on_click=self._zoom_out).props('flat dense size=sm').tooltip('缩小')
                        ui.button(icon='add', on_click=self._zoom_in).props('flat dense size=sm').tooltip('放大')
                        ui.button('Reset', on_click=self._zoom_reset).props('flat dense size=sm').tooltip('重置缩放')
                    
                    self.zoom_slider = ui.slider(
                        min=1, max=20, step=0.01, value=1,
                        on_change=self._on_zoom_slider
                    ).classes('w-full')
                
                ui.separator()
                
                # Box Actions (one-click operations)
                ui.label('Box Actions').classes('text-sm font-bold text-gray-700')
                with ui.column().classes('gap-2 w-full'):
                    ui.button(
                        'Swap Editable',
                        icon='swap_horiz',
                        on_click=self._on_swap_editable
                    ).classes('w-full text-xs').props('outline dense').tooltip(
                        '交换可编辑状态：可编辑框变为参考框，参考框变为可编辑框'
                    )
                    
                    ui.button(
                        'Clear Editable',
                        icon='delete_sweep',
                        on_click=self._on_clear_editable
                    ).classes('w-full text-xs').props('outline dense color=negative').tooltip(
                        '删除所有可编辑框'
                    )
                    
                    ui.button(
                        'Activate Reference',
                        icon='check_circle',
                        on_click=self._on_activate_reference
                    ).classes('w-full text-xs').props('outline dense color=positive').tooltip(
                        '将所有参考框变为可编辑'
                    )
                    
                    # Annotation cleaning checkbox
                    with ui.row().classes('items-center gap-2').tooltip('清理标注：删除重复标注、大框框小框和相似标注'):
                        self.clean_annotations_checkbox = ui.checkbox(
                            'Clean Annotations',
                            value=False,
                            on_change=self._on_clean_annotations_toggle
                        ).classes('text-sm')
                    
                    # Clean Annotations threshold settings
                    ui.label('Threshold').classes('text-xs text-gray-500')
                    with ui.row().classes('items-center gap-2'):
                        self.clean_threshold = ui.radio(
                            ['Low', 'Medium', 'High'],
                            value='Medium',
                            on_change=self._on_clean_threshold_change
                        ).props('inline').classes('text-xs')
                
                ui.separator()
                
                # Save Controls
                ui.label('Save Controls').classes('text-sm font-bold text-gray-700')
                with ui.column().classes('gap-2 w-full'):
                    self.extend_gt_to_next_checkbox = ui.checkbox(
                        'Extend GT to Next',
                        value=False,
                        on_change=self._on_extend_gt_to_next_toggle,
                    ).classes('text-sm').tooltip('将当前标注延伸到下一帧')
                    self.extend_prefer_previous_on_overlap_checkbox = ui.checkbox(
                        'Prefer Previous on Overlap',
                        value=False,
                        on_change=lambda e: setattr(self, 'extend_prefer_previous_on_overlap_enabled', bool(e.value)),
                    ).classes('text-sm').tooltip(
                        '当启用时：同类组(0/2、1/3)且 IoU>0.2 的重叠情况下，优先采纳前一帧（删除后一帧重叠可编辑框）'
                    )
                    # Sync initial enabled/disabled state for sub-toggle
                    self._set_extend_gt_to_next_enabled(self.extend_gt_to_next_enabled)
                    self.auto_save_checkbox = ui.checkbox(
                        'Auto Save',
                        value=True,
                        on_change=lambda e: setattr(self, 'auto_save_enabled', e.value)
                    ).classes('text-sm').tooltip('自动保存标注')
                    self.save_unmodified_checkbox = ui.checkbox(
                        'Save Unmodified',
                        value=True,
                        on_change=lambda e: setattr(self, 'save_unmodified_enabled', e.value)
                    ).classes('text-sm').tooltip('保存未修改的标注')
                    
                    self.save_button = ui.button(
                        'Save',
                        icon='save',
                        on_click=self._on_save
                    ).classes('w-full').tooltip('保存标注')
                
                ui.separator()
                
                # Navigation
                ui.button(
                    'Go Back to Analysis',
                    icon='analytics',
                    on_click=self._on_back
                ).classes('w-full').props('outline').tooltip('返回分析页面')
                
                # Keyboard shortcuts info
                with ui.expansion('Keyboard Shortcuts', icon='keyboard').classes('w-full text-xs'):
                    with ui.column().classes('gap-1 text-gray-600'):
                        ui.label('[ / ] - Prev/Next image')
                        ui.label('y - Toggle Extend GT to Next')
                        ui.label('u - Toggle Prefer Previous')
                        ui.label('· - Cycle selection')
                        ui.label('Del/Backspace - Delete box')
                        ui.label('Arrow keys - Move box')
                        ui.label('1/2/3/4 - Set class 0/1/2/3')
                        ui.label('q - Toggle Show GT')
                        ui.label('w - Toggle Show Pred')
                        ui.label('e - Swap Editable')
                        ui.label('r - Clear Editable')
                        ui.label('t - Activate Reference')
                        ui.label('Ctrl+Z/Y - Undo/Redo')
                        ui.label('= / - - Zoom in/out')
                        ui.label('0 - Reset zoom')
    
    def _load_current_image(self):
        """Load the current image from the queue"""
        if not app_state.annotation_queue:
            return
        
        idx = app_state.current_annotation_index
        if idx < 0 or idx >= len(app_state.annotation_queue):
            idx = 0
            app_state.current_annotation_index = 0
        
        item = app_state.annotation_queue[idx]
        
        # Update header info
        self._update_header_info(item, idx)
        
        # Load image
        img_path = Path(app_state.config.images_path) / item.image_path
        if not img_path.exists():
            ui.notify(f'Image not found: {img_path}', type='negative')
            return
        
        self.annotator.load_image(str(img_path))
        
        # Load boxes
        gt_boxes, pred_boxes = self._load_boxes(item)
        self._extend_backup_path_for_current = None

        # If Extend will be applied to this frame, backup original boxes first
        if self._pending_extend_gt_to_next:
            try:
                backup_path = self._get_extend_backup_path(item)
                if backup_path is not None:
                    self._save_extend_backup(backup_path, gt_boxes, pred_boxes)
                    self._extend_backup_path_for_current = backup_path
            except Exception as ex:
                logger.warning(f'Failed to write extend backup: {ex}')

        # Apply "Extend GT to Next" (only when entering next frame)
        injected_count = 0
        if self._pending_extend_gt_to_next:
            try:
                img_w = int(self.annotator.image_width)
                img_h = int(self.annotator.image_height)
                gt_boxes, pred_boxes, injected_count = apply_extend_gt_to_next(
                    gt_boxes,
                    pred_boxes,
                    self._cached_editable_yolo,
                    img_w,
                    img_h,
                    prefer_previous_on_overlap=self.extend_prefer_previous_on_overlap_enabled,
                )
            finally:
                # One-shot behavior: only apply to the immediate next frame
                self._pending_extend_gt_to_next = False
                self._cached_editable_yolo = []

        self.annotator.load_boxes(gt_boxes, pred_boxes)
        self.current_gt_boxes = self.annotator.get_gt_boxes()
        # If Extend injected/cleared boxes, treat the frame as modified so Auto Save works
        # even when "Save Unmodified" is disabled.
        self.boxes_modified = injected_count > 0
        
        # Check if current image has any boxes
        has_boxes = len(gt_boxes) > 0 or len(pred_boxes) > 0
        
        # Auto-focus on boxes after loading image and boxes
        # auto_focus_boxes() will call on_zoom_change callback to update UI
        if self.annotator.image_width > 0 and self.annotator.image_height > 0:
            if has_boxes:
                # Auto focus on boxes
                if self.auto_focus_enabled:
                    self.annotator.auto_focus_boxes()
                else:
                    self.annotator.reset_zoom()
                # Image has boxes, save the view state after auto focus
                self.previous_view_state = self.annotator.get_view_state()
            else:
                # Image has no boxes: do not inherit high-zoom state from previous image.
                # This prevents empty-edge regions (corners/sides) from being centered after restore.
                if self.previous_view_state and float(self.previous_view_state.get('zoom', 1.0)) <= 1.0:
                    self.annotator.set_view_state(self.previous_view_state)
                else:
                    self.annotator.reset_zoom()
        
        # Update navigation buttons
        self._update_nav_buttons()
        
        # Update box list panel
        self._update_box_list()
        
        # Reset annotation cleaning state but keep checkbox state
        self.cleared_boxes = []
        self.cleaning_applied = False
        
        # If Clean Annotations is checked, re-apply cleaning
        if self.clean_annotations_checkbox and self.clean_annotations_checkbox.value:
            self._clean_annotations()

    def _get_extend_backup_path(self, item: IssueItem) -> Optional[Path]:
        """Compute the backup file path for the current frame (not using *_tmp.txt)."""
        out_root = (app_state.config.output_path or '').strip()
        if not out_root:
            return None
        out_path = Path(out_root)
        rel = Path(item.image_path)
        # Store in a hidden subfolder to avoid mixing with output labels and *_tmp.txt
        backup_root = out_path / '.refiner_extend_backups'
        # Use the same relative structure, but JSON files
        return (backup_root / rel).with_suffix('.json')

    def _save_extend_backup(self, path: Path, gt_boxes: List[BBox], pred_boxes: List[BBox]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = boxes_to_backup_payload(gt_boxes, pred_boxes)
        path.write_text(dumps_backup_payload(payload), encoding='utf-8')

    def _restore_extend_backup_for_current(self) -> bool:
        """Restore current frame boxes from backup (if present). Returns whether restored."""
        if not self._extend_backup_path_for_current:
            return False
        if not self.annotator:
            return False
        path = self._extend_backup_path_for_current
        if not path.exists():
            return False
        payload = loads_backup_payload(path.read_text(encoding='utf-8'))
        gt_boxes, pred_boxes = boxes_from_backup_payload(payload)
        self.annotator.load_boxes(gt_boxes, pred_boxes)
        self.current_gt_boxes = self.annotator.get_gt_boxes()
        self.boxes_modified = False
        self._update_box_list()
        return True
    
    def _update_header_info(self, item: IssueItem, idx: int):
        """Update header labels"""
        total = len(app_state.annotation_queue)
        
        self.title_label.text = item.image_path
        self.progress_label.text = f'({idx + 1}/{total})'
        
        # Issue type with color
        issue_type = item.issue_type.value.replace('_', ' ').title()
        color_map = {
            IssueType.OVERLOOKED: 'orange',
            IssueType.SWAPPED: 'red',
            IssueType.BAD_LOCATED: 'purple'
        }
        color = color_map.get(item.issue_type, 'gray')
        self.issue_info_label.text = f'{issue_type} | Score: {item.score:.4f}'
        self.issue_info_label.classes(remove='text-orange-600 text-red-600 text-purple-600')
        self.issue_info_label.classes(add=f'text-{color}-600 font-medium')

    def _on_auto_focus_change(self, e) -> None:
        """Handle auto focus checkbox change."""
        self.auto_focus_enabled = bool(e.value)
        if not self.annotator:
            return
        if self.annotator.image_width <= 0 or self.annotator.image_height <= 0:
            return
        if self.auto_focus_enabled:
            self.annotator.auto_focus_boxes()
        else:
            self.annotator.reset_zoom()
    
    def _load_boxes(self, item: IssueItem) -> tuple:
        """Load GT and Pred boxes for an item"""
        rel_path = Path(item.image_path)
        
        # Image dimensions
        img_path = Path(app_state.config.images_path) / rel_path
        img_w, img_h = get_image_size(img_path)
        
        class_name_to_id = build_class_name_to_id(app_state.class_mapping)

        # GT labels - prefer output tmp/finalized labels, then fall back to source labels.
        output_path = Path(app_state.config.output_path) if (app_state.config.output_path or '').strip() else None
        key = rel_path.with_suffix('')

        gt_raw = []
        if output_path:
            output_tmp_path = resolve_label_path(output_path, key, include_tmp=True)
            output_label_path = resolve_label_path(output_path, key)
            if output_tmp_path and output_tmp_path.exists():
                gt_raw = read_label_file(output_tmp_path, img_w, img_h, has_confidence=False, class_name_to_id=class_name_to_id)
            elif output_label_path and output_label_path.exists():
                gt_raw = read_label_file(output_label_path, img_w, img_h, has_confidence=False, class_name_to_id=class_name_to_id)

        if not gt_raw:
            gt_labels_root = (app_state.config.gt_labels_path or '').strip()
            if gt_labels_root and Path(gt_labels_root).exists():
                gt_label_path = resolve_label_path(Path(gt_labels_root), key)
                gt_raw = read_label_file(gt_label_path, img_w, img_h, has_confidence=False, class_name_to_id=class_name_to_id)
        
        # Pred labels (only if configured)
        pred_raw = []
        pred_labels_root = (app_state.config.pred_labels_path or '').strip()
        if pred_labels_root and Path(pred_labels_root).exists():
            pred_label_path = resolve_label_path(Path(pred_labels_root), key)
            pred_raw = read_label_file(pred_label_path, img_w, img_h, has_confidence=True, class_name_to_id=class_name_to_id)
        
        # Convert to BBox objects
        # GT boxes: editable=True (can be edited)
        gt_boxes = []
        for i, box in enumerate(gt_raw):
            x1, y1, x2, y2 = box['bbox']
            gt_boxes.append(BBox(
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                class_id=box['class_id'],
                source=BoxSource.GT,
                visible=True,
                editable=True
            ))
        
        # Pred boxes: editable=False (reference only by default)
        pred_boxes = []
        for i, box in enumerate(pred_raw):
            x1, y1, x2, y2 = box['bbox']
            pred_boxes.append(BBox(
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                class_id=box['class_id'],
                source=BoxSource.PRED,
                visible=True,
                editable=False
            ))
        
        return gt_boxes, pred_boxes
    
    def _update_nav_buttons(self):
        """Update navigation button states"""
        idx = app_state.current_annotation_index
        total = len(app_state.annotation_queue)
        
        if idx <= 0:
            self.prev_button.disable()
        else:
            self.prev_button.enable()
        
        if idx >= total - 1:
            self.next_button.disable()
        else:
            self.next_button.enable()
    
    def _on_boxes_changed(self, boxes: List[BBox]):
        """Callback when GT boxes change"""
        self.current_gt_boxes = boxes
        self.boxes_modified = True
        # Update box list panel
        self._update_box_list()
    
    def _on_display_change(self, e=None):
        """Handle display option change from UI checkbox"""
        if self.annotator:
            self.annotator.set_display_options(
                self.show_gt_checkbox.value,
                self.show_pred_checkbox.value
            )
    
    def _on_display_change_from_annotator(self, show_gt: bool, show_pred: bool):
        """Handle display option change from annotator (e.g., keyboard shortcut)"""
        if self.show_gt_checkbox:
            self.show_gt_checkbox.value = show_gt
        if self.show_pred_checkbox:
            self.show_pred_checkbox.value = show_pred
    
    def _on_zoom_changed(self, zoom: float):
        """Handle zoom change from annotator"""
        self.zoom_label.text = f'{zoom:.2f}x'
        if self.zoom_slider:
            self.zoom_slider.value = zoom
    
    def _zoom_in(self):
        """Zoom in"""
        if self.annotator:
            self.annotator.zoom_in()
    
    def _zoom_out(self):
        """Zoom out"""
        if self.annotator:
            self.annotator.zoom_out()
    
    def _zoom_reset(self):
        """Reset zoom"""
        if self.annotator:
            self.annotator.reset_zoom()
    
    def _on_zoom_slider(self, e):
        """Handle zoom slider change"""
        if self.annotator:
            self.annotator.set_zoom(e.value)
    
    def _on_swap_editable(self):
        """Swap editable status of all boxes"""
        if self.annotator:
            self.annotator.swap_editable()
            self._update_box_list()
            ui.notify('Swapped editable status', type='info', position='bottom-right', timeout=1000)
    
    def _on_clear_editable(self):
        """Delete all editable boxes"""
        if self.annotator:
            self.annotator.clear_editable()
            self._update_box_list()
            ui.notify('Cleared all editable boxes', type='warning', position='bottom-right', timeout=1000)
    
    def _on_activate_reference(self):
        """Make all reference boxes editable"""
        if self.annotator:
            self.annotator.activate_reference()
            self._update_box_list()
            ui.notify('Activated reference boxes', type='positive', position='bottom-right', timeout=1000)
    
    def _on_clean_annotations_toggle(self, e):
        """Handle annotation cleaning checkbox toggle"""
        if not self.annotator:
            return
        
        if e.value:
            # Apply cleaning
            self._clean_annotations()
        else:
            # Revert cleaning
            self._revert_cleaning()
    
    def _on_clean_threshold_change(self, e):
        """Handle clean threshold change"""
        if self.clean_annotations_checkbox and self.clean_annotations_checkbox.value:
            # Re-apply cleaning with new threshold
            self._clean_annotations()
    
    def _clean_annotations(self):
        """Clean annotations by removing duplicate, large containing, and similar boxes"""
        if not self.annotator:
            return
        
        # Get current boxes
        gt_boxes = self.annotator.gt_boxes.copy()
        pred_boxes = self.annotator.pred_boxes.copy()
        
        # Process boxes and get removed box IDs
        cleaned_gt, cleaned_pred, removed_box_ids = self._process_boxes(gt_boxes, pred_boxes)
        
        # Store removed box IDs for revert
        self.cleared_boxes = removed_box_ids
        
        # Update annotator with cleaned boxes
        self.annotator.gt_boxes = cleaned_gt
        self.annotator.pred_boxes = cleaned_pred
        self.annotator._update_display()
        
        # Update box list
        self._update_box_list()
        
        # Update clean status label on image
        removed_count = len(removed_box_ids)
        if hasattr(self.annotator, 'clean_status_label') and self.annotator.clean_status_label:
            if removed_count > 0:
                self.annotator.clean_status_label.text = f'Clean Annotations：{removed_count}'
                self.annotator.clean_status_label.style('display: block;')
            else:
                self.annotator.clean_status_label.style('display: none;')
        
        # Mark cleaning as applied
        self.cleaning_applied = True
        
        # Show notification
        ui.notify(f'Cleaned annotations: removed {removed_count} boxes', type='info', position='bottom-right', timeout=1500)
    
    def _revert_cleaning(self):
        """Revert annotation cleaning - only restore removed boxes"""
        if not self.annotator or not self.cleaning_applied:
            return
        
        # Restore only the removed boxes
        if self.cleared_boxes:
            # Get current boxes
            current_gt = self.annotator.gt_boxes
            current_pred = self.annotator.pred_boxes
            
            # Add back removed boxes
            for box in self.cleared_boxes:
                if box.source.value == 'gt':
                    current_gt.append(box)
                else:
                    current_pred.append(box)
            
            # Update annotator
            self.annotator.gt_boxes = current_gt
            self.annotator.pred_boxes = current_pred
            self.annotator._update_display()
            
            # Update box list
            self._update_box_list()
            
            # Hide clean status label
            if hasattr(self.annotator, 'clean_status_label') and self.annotator.clean_status_label:
                self.annotator.clean_status_label.style('display: none;')
            
            # Reset state
            self.cleaning_applied = False
            self.cleared_boxes = []
            
            # Show notification
            ui.notify('Reverted annotation cleaning', type='info', position='bottom-right', timeout=1500)
    
    def _process_boxes(self, gt_boxes, pred_boxes):
        """Process boxes to remove duplicate, large containing, and similar boxes"""
        import copy
        
        # Get threshold setting
        threshold = getattr(self, 'clean_threshold', None)
        threshold_value = threshold.value if threshold else 'Medium'
        
        # Set thresholds based on selected level
        if threshold_value == 'Low':
            duplicate_iou_threshold = 0.85
            similar_iou_threshold = 0.85
            boundary_threshold = 10
            min_contained_boxes = 2
        elif threshold_value == 'High':
            duplicate_iou_threshold = 0.45
            similar_iou_threshold = 0.45
            boundary_threshold = 50
            min_contained_boxes = 1
        else:  # Medium
            duplicate_iou_threshold = 0.7
            similar_iou_threshold = 0.7
            boundary_threshold = 30
            min_contained_boxes = 2
        
        # Combine all boxes
        all_boxes = copy.deepcopy(gt_boxes) + copy.deepcopy(pred_boxes)
        removed_boxes = []  # Store removed boxes for revert
        
        # Mark boxes to remove
        to_remove = set()
        
        # 1. Remove duplicate boxes (keep larger one)
        for i in range(len(all_boxes)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(all_boxes)):
                if j in to_remove:
                    continue
                
                box1 = all_boxes[i]
                box2 = all_boxes[j]
                
                # Calculate IoU
                iou = self._calculate_iou(
                    (box1.x, box1.y, box1.x + box1.w, box1.y + box1.h),
                    (box2.x, box2.y, box2.x + box2.w, box2.y + box2.h)
                )
                
                if iou > duplicate_iou_threshold:
                    # Remove larger box (keep smaller one)
                    area1 = box1.w * box1.h
                    area2 = box2.w * box2.h
                    if area1 > area2:
                        to_remove.add(i)
                        removed_boxes.append(box1)
                    else:
                        to_remove.add(j)
                        removed_boxes.append(box2)
        
        # Filter out duplicate boxes
        filtered_boxes = [box for i, box in enumerate(all_boxes) if i not in to_remove]
        
        # 2. Remove large boxes that contain smaller boxes based on strictness level
        to_remove_large = set()
        for i in range(len(filtered_boxes)):
            if i in to_remove_large:
                continue
            
            large_box = filtered_boxes[i]
            large_box_coords = (large_box.x, large_box.y, large_box.x + large_box.w, large_box.y + large_box.h)
            
            # Count how many smaller boxes are contained in this large box
            contained_count = 0
            for j in range(len(filtered_boxes)):
                if i == j or j in to_remove_large:
                    continue
                
                small_box = filtered_boxes[j]
                small_box_coords = (small_box.x, small_box.y, small_box.x + small_box.w, small_box.y + small_box.h)
                
                if self._is_box_inside(small_box_coords, large_box_coords):
                    contained_count += 1
            
            # Remove large box if it contains enough smaller boxes based on strictness level
            if contained_count >= min_contained_boxes:
                to_remove_large.add(i)
                removed_boxes.append(large_box)
        
        # Filter out large boxes
        filtered_boxes = [box for i, box in enumerate(filtered_boxes) if i not in to_remove_large]
        
        # 3. Remove different class boxes with similar positions
        to_remove_similar = set()
        for i in range(len(filtered_boxes)):
            if i in to_remove_similar:
                continue
            for j in range(i + 1, len(filtered_boxes)):
                if j in to_remove_similar:
                    continue
                
                box1 = filtered_boxes[i]
                box2 = filtered_boxes[j]
                
                # Check if different classes
                if box1.class_id != box2.class_id:
                    # Calculate IoU
                    iou = self._calculate_iou(
                        (box1.x, box1.y, box1.x + box1.w, box1.y + box1.h),
                        (box2.x, box2.y, box2.x + box2.w, box2.y + box2.h)
                    )
                    
                    # Calculate boundary differences
                    diffs = [
                        abs(box1.x - box2.x),
                        abs(box1.y - box2.y),
                        abs((box1.x + box1.w) - (box2.x + box2.w)),
                        abs((box1.y + box1.h) - (box2.y + box2.h))
                    ]
                    
                    # Check if similar
                    if iou > similar_iou_threshold and self._is_boundary_similar(diffs, boundary_threshold):
                        to_remove_similar.add(i)
                        to_remove_similar.add(j)
                        removed_boxes.append(box1)
                        removed_boxes.append(box2)
        
        # Filter out similar boxes
        filtered_boxes = [box for i, box in enumerate(filtered_boxes) if i not in to_remove_similar]
        
        # Separate back into GT and Pred boxes
        cleaned_gt = [box for box in filtered_boxes if getattr(box, 'source', '').value == 'gt' or getattr(box, 'source', '') == 'gt']
        cleaned_pred = [box for box in filtered_boxes if getattr(box, 'source', '').value == 'pred' or getattr(box, 'source', '') == 'pred']
        
        return cleaned_gt, cleaned_pred, removed_boxes
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _is_box_inside(self, inner_box, outer_box):
        """Check if inner box is inside outer box"""
        return (
            inner_box[0] >= outer_box[0] and
            inner_box[1] >= outer_box[1] and
            inner_box[2] <= outer_box[2] and
            inner_box[3] <= outer_box[3]
        )
    
    def _is_boundary_similar(self, diffs, boundary_threshold):
        """Check if boundary differences are similar
        
        Args:
            diffs: List of absolute differences [xmin_diff, ymin_diff, xmax_diff, ymax_diff]
            boundary_threshold: Maximum allowed difference for similarity
        
        Returns:
            True if any two differences are within the threshold
        """
        # Check if any two sides are within the boundary threshold
        count_within_threshold = sum(1 for d in diffs if d <= boundary_threshold)
        return count_within_threshold >= 2
    
    def _on_save(self):
        """Save current annotations
        
        Only saves boxes where editable=True (regardless of source GT/Pred)
        """
        if not app_state.annotation_queue or not self.annotator:
            return
        
        item = app_state.annotation_queue[app_state.current_annotation_index]
        rel_path = Path(item.image_path)
        
        # Get image dimensions
        img_path = Path(app_state.config.images_path) / rel_path
        img_w, img_h = get_image_size(img_path)
        
        # Get all boxes and filter for editable ones only
        all_boxes = self.annotator.get_all_boxes()
        
        # Convert editable BBox to dict format for save_tmp_annotation
        boxes_dict = []
        for box in all_boxes:
            if getattr(box, 'editable', True):
                boxes_dict.append({
                    'class_id': box.class_id,
                    'bbox': [box.x, box.y, box.x + box.w, box.y + box.h]
                })
        
        # Save to tmp file in output path
        save_tmp_annotation(
            app_state.config.output_path,
            item.image_path,
            boxes_dict,
            img_w, img_h,
            gt_labels_path=app_state.config.gt_labels_path,
            class_id_to_name=build_class_id_to_name(app_state.class_mapping),
        )
        
        self.boxes_modified = False
        ui.notify('Saved', type='positive', position='bottom-right', timeout=1000)
    
    def _on_prev(self):
        """Go to previous image"""
        if app_state.current_annotation_index > 0:
            self._before_navigate()
            app_state.current_annotation_index -= 1
            self._load_current_image()
    
    def _on_next(self):
        """Go to next image"""
        if app_state.current_annotation_index < len(app_state.annotation_queue) - 1:
            self._prepare_extend_gt_to_next_if_needed()
            self._before_navigate()
            app_state.current_annotation_index += 1
            self._load_current_image()
    
    def _before_navigate(self):
        """Called before navigating to another image"""
        if self._should_auto_save():
            self._on_save()

    def _should_auto_save(self) -> bool:
        """Return whether auto-save should run on navigation"""
        return self.auto_save_enabled and (self.boxes_modified or self.save_unmodified_enabled)

    def _prepare_extend_gt_to_next_if_needed(self) -> None:
        """Cache current editable boxes for applying to the next frame."""
        if not self.extend_gt_to_next_enabled:
            self._pending_extend_gt_to_next = False
            self._cached_editable_yolo = []
            return
        if not self.annotator:
            self._pending_extend_gt_to_next = False
            self._cached_editable_yolo = []
            return
        if self.annotator.image_width <= 0 or self.annotator.image_height <= 0:
            self._pending_extend_gt_to_next = False
            self._cached_editable_yolo = []
            return

        all_boxes = self.annotator.get_all_boxes()
        self._cached_editable_yolo = editable_boxes_to_yolo(
            all_boxes,
            int(self.annotator.image_width),
            int(self.annotator.image_height),
        )
        self._pending_extend_gt_to_next = True
        logger.info(f'Extend GT to Next: cached {len(self._cached_editable_yolo)} editable boxes for next frame')
    
    def _handle_page_keys(self, e):
        """Handle page-level keyboard events"""
        if not e.action.keydown:
            return
        
        key_name = e.key.name if hasattr(e.key, 'name') else str(e.key)
        
        # [ for previous
        if key_name == 'BracketLeft' or key_name == '[':
            self._on_prev()
            return
        
        # ] for next
        if key_name == 'BracketRight' or key_name == ']':
            self._on_next()
            return

        # y toggles "Extend GT to Next"
        if not e.modifiers.ctrl and not e.modifiers.alt and not e.modifiers.shift:
            if str(key_name).lower() == 'y':
                self._set_extend_gt_to_next_enabled(not self.extend_gt_to_next_enabled)
                ui.notify(
                    f'Extend GT to Next: {"ON" if self.extend_gt_to_next_enabled else "OFF"}',
                    type='info',
                    position='bottom-right',
                    timeout=1200,
                )
                return
            
            # u toggles "Prefer Previous on Overlap" (only when Extend is enabled)
            if str(key_name).lower() == 'u':
                if self.extend_gt_to_next_enabled:
                    self.extend_prefer_previous_on_overlap_enabled = not self.extend_prefer_previous_on_overlap_enabled
                    if self.extend_prefer_previous_on_overlap_checkbox is not None:
                        self.extend_prefer_previous_on_overlap_checkbox.value = self.extend_prefer_previous_on_overlap_enabled
                    ui.notify(
                        f'Prefer Previous on Overlap: {"ON" if self.extend_prefer_previous_on_overlap_enabled else "OFF"}',
                        type='info',
                        position='bottom-right',
                        timeout=1200,
                    )
                else:
                    ui.notify(
                        'Prefer Previous on Overlap requires Extend GT to Next to be enabled',
                        type='warning',
                        position='bottom-right',
                        timeout=1500,
                    )
                return

    def _on_extend_gt_to_next_toggle(self, e) -> None:
        """Handle Extend GT to Next checkbox toggle from UI."""
        self._set_extend_gt_to_next_enabled(bool(e.value))

    def _set_extend_gt_to_next_enabled(self, enabled: bool) -> None:
        """Set Extend GT to Next and sync dependent sub-toggle UI/state."""
        was_enabled = bool(self.extend_gt_to_next_enabled)
        self.extend_gt_to_next_enabled = bool(enabled)
        if self.extend_gt_to_next_checkbox is not None:
            self.extend_gt_to_next_checkbox.value = self.extend_gt_to_next_enabled

        # Sub-toggle is only meaningful when extend is enabled
        if not self.extend_gt_to_next_enabled:
            self.extend_prefer_previous_on_overlap_enabled = False
            if self.extend_prefer_previous_on_overlap_checkbox is not None:
                self.extend_prefer_previous_on_overlap_checkbox.value = False

        # Enable/disable sub-toggle UI
        if self.extend_prefer_previous_on_overlap_checkbox is not None:
            if self.extend_gt_to_next_enabled:
                self.extend_prefer_previous_on_overlap_checkbox.enable()
            else:
                self.extend_prefer_previous_on_overlap_checkbox.disable()

        # If user turns off Extend while staying on a frame that was extended,
        # restore the original boxes from backup immediately.
        if was_enabled and (not self.extend_gt_to_next_enabled):
            try:
                restored = self._restore_extend_backup_for_current()
                if restored:
                    ui.notify('Restored original annotations for this frame', type='info', position='bottom-right', timeout=1500)
            except Exception as ex:
                logger.warning(f'Failed to restore extend backup: {ex}')
    
    async def _on_back(self):
        """Handle back button click"""
        # Scanning output dir can be expensive; do it off the event loop.
        try:
            loop = asyncio.get_event_loop()
            tmp_files = await loop.run_in_executor(
                None, lambda: get_tmp_files(app_state.config.output_path)
            )
        except Exception as ex:
            self._safe_notify(f'读取临时文件失败: {ex}', type='negative')
            logger.error(f'Failed to list tmp files: {ex}')
            return

        if not tmp_files:
            # No modifications, go directly back
            self._safe_navigate('/')
            return

        # Show confirmation dialog
        self._show_confirm_dialog(len(tmp_files), tmp_files)
    
    def _show_confirm_dialog(self, modified_count: int, tmp_files: List[str]):
        """Show confirmation dialog for changes"""
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            with ui.column().classes('w-full gap-4 p-4'):
                ui.label('Confirm Changes').classes('text-xl font-bold')
                
                ui.label(f'You have modified {modified_count} image(s).').classes('text-gray-600')
                ui.label('How would you like to proceed?').classes('text-gray-600')
                
                ui.separator()

                async def _keep_and_back() -> None:
                    await self._confirm_and_navigate(dialog, tmp_files, True)

                async def _discard_and_back() -> None:
                    await self._confirm_and_navigate(dialog, tmp_files, False)
                
                # Yes - Keep Changes
                with ui.button(
                    on_click=_keep_and_back
                ).classes('w-full').props('color=positive'):
                    with ui.column().classes('items-center gap-0'):
                        ui.label('Yes - Keep Changes').classes('font-medium')
                        ui.label('(Overwrite original GT with tmp files)').classes('text-xs opacity-70')
                
                # No - Discard Changes
                with ui.button(
                    on_click=_discard_and_back
                ).classes('w-full').props('color=negative'):
                    with ui.column().classes('items-center gap-0'):
                        ui.label('No - Discard Changes').classes('font-medium')
                        ui.label('(Delete all tmp files)').classes('text-xs opacity-70')
                
                # Cancel
                ui.button(
                    'Cancel',
                    on_click=dialog.close
                ).classes('w-full').props('outline')
        
        dialog.open()
    
    async def _confirm_and_navigate(self, dialog, tmp_files: List[str], keep_changes: bool):
        """Confirm changes and navigate back without blocking the UI."""
        try:
            dialog.close()
        except Exception:
            pass

        # Show a lightweight "working" hint; avoid crashing if client disconnects.
        self._safe_notify('正在处理更改，请稍候...', type='info', timeout=2000)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: confirm_changes_for_tmp_files(
                    app_state.config.output_path,
                    tmp_files,
                    keep_changes=keep_changes,
                ),
            )

            if keep_changes:
                self._safe_notify('更改已保存', type='positive')
            else:
                self._safe_notify('已丢弃更改', type='info')
        except Exception as ex:
            self._safe_notify(f'处理更改失败: {ex}', type='negative', timeout=5000)
            logger.error(f'Error confirming changes: {ex}')
            return

        self._safe_navigate('/')

    def _safe_notify(self, message: str, type: str = 'info', timeout: int = 3000) -> None:
        """Notify if a client is available; otherwise ignore."""
        try:
            ui.notify(message, type=type, timeout=timeout)
        except RuntimeError:
            return

    def _safe_navigate(self, path: str) -> None:
        """Navigate if a client is available; otherwise ignore."""
        try:
            ui.navigate.to(path)
        except RuntimeError:
            return


def create_annotator():
    """Create annotator page"""
    page = AnnotatorPage()
    page.create()
    return page

