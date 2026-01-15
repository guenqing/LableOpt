"""
Annotator page - Interactive annotation editing interface
"""
from pathlib import Path
from typing import List, Optional
import logging

from nicegui import ui

from ..state import app_state
from ..models import BBox, BoxSource, IssueItem, IssueType
from ..core.yolo_utils import read_yolo_label, get_image_size
from ..core.file_manager import save_tmp_annotation, confirm_changes, get_tmp_files
from .components import InteractiveAnnotator

logger = logging.getLogger(__name__)


class AnnotatorPage:
    """Annotator page for editing bounding box annotations"""
    
    def __init__(self):
        # Auto save flag
        self.auto_save_enabled: bool = False
        
        # Current GT boxes (from annotator component)
        self.current_gt_boxes: List[BBox] = []
        
        # Track if current image has unsaved changes
        self.boxes_modified: bool = False
        
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
        self.class_select = None
        self.show_gt_checkbox = None
        self.show_pred_checkbox = None
        self.auto_save_checkbox = None
        
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
            
            # Main content area
            with ui.row().classes('flex-grow w-full'):
                # Left: Annotator area
                self._create_annotator_area()
                
                # Right: Control panel
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
            ui.button('Back to Dashboard', on_click=lambda: ui.navigate.to('/')).classes('mt-6')
    
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
    
    def _create_annotator_area(self):
        """Create the annotator component area"""
        with ui.column().classes('flex-grow p-4'):
            # Annotator container
            annotator_container = ui.column().classes('bg-white rounded-lg shadow p-2')
            
            # Create annotator
            self.annotator = InteractiveAnnotator(
                on_change=self._on_boxes_changed,
                on_zoom_change=self._on_zoom_changed
            )
            self.annotator.create_ui(annotator_container, fixed_width=900, fixed_height=600)
            
            # Navigation buttons below annotator
            with ui.row().classes('w-full justify-center gap-4 mt-4'):
                self.prev_button = ui.button(
                    'Prev', 
                    icon='chevron_left',
                    on_click=self._on_prev
                ).props('outline')
                
                self.next_button = ui.button(
                    'Next',
                    icon='chevron_right',
                    on_click=self._on_next
                ).props('outline icon-right')
    
    def _create_control_panel(self):
        """Create control panel on the right side"""
        with ui.card().classes('w-72 flex-shrink-0 control-panel m-4'):
            with ui.column().classes('w-full p-4 gap-4'):
                # Display Options
                ui.label('Display Options').classes('text-sm font-bold text-gray-700')
                with ui.column().classes('gap-1'):
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
                        ui.button(icon='remove', on_click=self._zoom_out).props('flat dense size=sm')
                        ui.button(icon='add', on_click=self._zoom_in).props('flat dense size=sm')
                        ui.button('Reset', on_click=self._zoom_reset).props('flat dense size=sm')
                    
                    self.zoom_slider = ui.slider(
                        min=1, max=10, step=1, value=1,
                        on_change=self._on_zoom_slider
                    ).classes('w-full')
                
                ui.separator()
                
                # Current Class
                ui.label('Current Class').classes('text-sm font-bold text-gray-700')
                self.class_select = ui.select(
                    options={0: '0', 1: '1', 2: '2', 3: '3'},
                    value=0,
                    on_change=self._on_class_change
                ).classes('w-full')
                
                ui.separator()
                
                # Save Controls
                ui.label('Save Controls').classes('text-sm font-bold text-gray-700')
                with ui.column().classes('gap-2 w-full'):
                    self.auto_save_checkbox = ui.checkbox(
                        'Auto Save',
                        value=False,
                        on_change=lambda e: setattr(self, 'auto_save_enabled', e.value)
                    ).classes('text-sm')
                    
                    self.save_button = ui.button(
                        'Save',
                        icon='save',
                        on_click=self._on_save
                    ).classes('w-full')
                
                ui.separator()
                
                # Navigation
                ui.button(
                    'Go Back to Analysis',
                    icon='analytics',
                    on_click=self._on_back
                ).classes('w-full').props('outline')
                
                # Keyboard shortcuts info
                with ui.expansion('Keyboard Shortcuts', icon='keyboard').classes('w-full text-xs'):
                    with ui.column().classes('gap-1 text-gray-600'):
                        ui.label('[ / ] - Prev/Next image')
                        ui.label('Tab - Cycle selection')
                        ui.label('Del/Backspace - Delete box')
                        ui.label('Arrow keys - Move box')
                        ui.label('q/w/e/r - Set class 0/1/2/3')
                        ui.label('Ctrl+Z/Y - Undo/Redo')
    
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
        self.annotator.load_boxes(gt_boxes, pred_boxes)
        self.current_gt_boxes = self.annotator.get_gt_boxes()
        self.boxes_modified = False
        
        # Update navigation buttons
        self._update_nav_buttons()
        
        # Reset zoom display
        self._on_zoom_changed(1.0)
    
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
    
    def _load_boxes(self, item: IssueItem) -> tuple:
        """Load GT and Pred boxes for an item"""
        rel_path = Path(item.image_path)
        
        # Image dimensions
        img_path = Path(app_state.config.images_path) / rel_path
        img_w, img_h = get_image_size(img_path)
        
        # GT labels - check for tmp file first
        gt_label_path = Path(app_state.config.gt_labels_path) / rel_path.with_suffix('.txt')
        tmp_label_path = gt_label_path.parent / f'{gt_label_path.stem}_tmp.txt'
        
        if tmp_label_path.exists():
            # Load from tmp file
            gt_raw = read_yolo_label(tmp_label_path, img_w, img_h, has_confidence=False)
        else:
            gt_raw = read_yolo_label(gt_label_path, img_w, img_h, has_confidence=False)
        
        # Pred labels
        pred_label_path = Path(app_state.config.pred_labels_path) / rel_path.with_suffix('.txt')
        pred_raw = read_yolo_label(pred_label_path, img_w, img_h, has_confidence=True)
        
        # Convert to BBox objects
        gt_boxes = []
        for i, box in enumerate(gt_raw):
            x1, y1, x2, y2 = box['bbox']
            gt_boxes.append(BBox(
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                class_id=box['class_id'],
                source=BoxSource.GT
            ))
        
        pred_boxes = []
        for i, box in enumerate(pred_raw):
            x1, y1, x2, y2 = box['bbox']
            pred_boxes.append(BBox(
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                class_id=box['class_id'],
                source=BoxSource.PRED
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
    
    def _on_display_change(self, e=None):
        """Handle display option change"""
        if self.annotator:
            self.annotator.set_display_options(
                self.show_gt_checkbox.value,
                self.show_pred_checkbox.value
            )
    
    def _on_zoom_changed(self, zoom: float):
        """Handle zoom change from annotator"""
        self.zoom_label.text = f'{int(zoom)}x'
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
    
    def _on_class_change(self, e):
        """Handle class selection change"""
        if self.annotator:
            self.annotator.set_current_class(e.value)
    
    def _on_save(self):
        """Save current annotations"""
        if not app_state.annotation_queue:
            return
        
        item = app_state.annotation_queue[app_state.current_annotation_index]
        rel_path = Path(item.image_path)
        
        # Get image dimensions
        img_path = Path(app_state.config.images_path) / rel_path
        img_w, img_h = get_image_size(img_path)
        
        # Convert BBox to dict format for save_tmp_annotation
        boxes_dict = []
        for box in self.current_gt_boxes:
            boxes_dict.append({
                'class_id': box.class_id,
                'bbox': [box.x, box.y, box.x + box.w, box.y + box.h]
            })
        
        # Save to tmp file
        save_tmp_annotation(
            app_state.config.gt_labels_path,
            item.image_path,
            boxes_dict,
            img_w, img_h
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
            self._before_navigate()
            app_state.current_annotation_index += 1
            self._load_current_image()
    
    def _before_navigate(self):
        """Called before navigating to another image"""
        if self.auto_save_enabled and self.boxes_modified:
            self._on_save()
    
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
    
    def _on_back(self):
        """Handle back button click"""
        # Check for any tmp files
        tmp_files = get_tmp_files(app_state.config.gt_labels_path)
        
        if not tmp_files:
            # No modifications, go directly back
            ui.navigate.to('/')
            return
        
        # Show confirmation dialog
        self._show_confirm_dialog(len(tmp_files))
    
    def _show_confirm_dialog(self, modified_count: int):
        """Show confirmation dialog for changes"""
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            with ui.column().classes('w-full gap-4 p-4'):
                ui.label('Confirm Changes').classes('text-xl font-bold')
                
                ui.label(f'You have modified {modified_count} image(s).').classes('text-gray-600')
                ui.label('How would you like to proceed?').classes('text-gray-600')
                
                ui.separator()
                
                # Yes - Keep Changes
                with ui.button(
                    on_click=lambda: self._confirm_and_navigate(dialog, True)
                ).classes('w-full').props('color=positive'):
                    with ui.column().classes('items-center gap-0'):
                        ui.label('Yes - Keep Changes').classes('font-medium')
                        ui.label('(Overwrite original GT with tmp files)').classes('text-xs opacity-70')
                
                # No - Discard Changes
                with ui.button(
                    on_click=lambda: self._confirm_and_navigate(dialog, False)
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
    
    def _confirm_and_navigate(self, dialog, keep_changes: bool):
        """Confirm changes and navigate back"""
        dialog.close()
        
        try:
            confirm_changes(app_state.config.gt_labels_path, keep_changes=keep_changes)
            
            if keep_changes:
                ui.notify('Changes saved successfully', type='positive')
            else:
                ui.notify('Changes discarded', type='info')
        except Exception as ex:
            ui.notify(f'Error: {ex}', type='negative')
            logger.error(f'Error confirming changes: {ex}')
            return
        
        ui.navigate.to('/')


def create_annotator():
    """Create annotator page"""
    page = AnnotatorPage()
    page.create()
    return page
