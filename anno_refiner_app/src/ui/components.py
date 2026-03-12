"""
Interactive Annotator Component for NiceGUI
Provides interactive bounding box editing functionality using ui.interactive_image
"""
from typing import List, Callable, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from copy import deepcopy
import logging

from nicegui import ui

from ..models import BBox, BoxSource
from ..core.yolo_utils import get_image_size

logger = logging.getLogger(__name__)


@dataclass
class HistoryState:
    """Snapshot of boxes state for undo/redo (includes visible and editable)"""
    gt_boxes: List[BBox]
    pred_boxes: List[BBox]
    selected_id: Optional[str] = None


class InteractiveAnnotator:
    """Interactive annotation component for bounding box editing"""
    
    # Color configuration
    COLORS = {
        'gt': {'normal': '#22c55e', 'selected': '#ef4444'},      # green/red
        'pred': {'normal': '#3b82f6', 'selected': '#ef4444'},    # blue/red
    }
    
    # Handle configuration
    HANDLE_SIZE = 8
    HANDLE_POSITIONS = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']
    
    # Constraints
    MIN_BOX_SIZE = 5  # Minimum box dimension in pixels
    MAX_HISTORY = 50  # Maximum undo history
    
    # Zoom step for button-based zooming
    ZOOM_STEP = 0.05  # Smaller step for smoother zooming
    
    def __init__(self, on_change: Callable[[List[BBox]], None] = None,
                 on_zoom_change: Callable[[float], None] = None,
                 on_display_change: Callable[[bool, bool], None] = None):
        """
        Args:
            on_change: Callback when GT boxes change
            on_zoom_change: Callback when zoom level changes
            on_display_change: Callback when display options change (show_gt, show_pred)
        """
        self.on_change = on_change
        self.on_zoom_change = on_zoom_change
        self.on_display_change = on_display_change
        
        # State
        self.image_path: Optional[str] = None
        self.image_width: int = 0
        self.image_height: int = 0
        
        self.gt_boxes: List[BBox] = []
        self.pred_boxes: List[BBox] = []
        
        self.show_gt: bool = True
        self.show_pred: bool = True
        
        self.selected_box_id: Optional[str] = None
        self.current_class: int = 0
        
        # Zoom and pan state
        self.zoom: float = 1.0  # Current zoom level
        self.pan_x: float = 0.0  # Pan offset X (in image pixels)
        self.pan_y: float = 0.0  # Pan offset Y (in image pixels)
        
        # Interaction state
        self.drag_mode: Optional[str] = None  # 'move', 'resize', 'create', 'pan'
        self.drag_handle: Optional[str] = None  # 'nw', 'n', etc.
        self.drag_start_x: float = 0
        self.drag_start_y: float = 0
        self.drag_box_start: Optional[Dict] = None  # Original box state when drag started
        self.pan_start_x: float = 0  # Pan start position
        self.pan_start_y: float = 0
        
        # History for undo/redo
        self.history: List[HistoryState] = []
        self.history_index: int = -1
        
        # UI components
        self.image_component = None
        self.keyboard = None
        self.container = None
        
    def create_ui(self, container, fixed_width: int = 800, fixed_height: int = 600,
                  navigator_container=None) -> None:
        """Create UI components in the specified container
        
        Args:
            container: Parent container for Viewer (main image area)
            fixed_width: Fixed width for the image area in pixels
            fixed_height: Fixed height for the image area in pixels
            navigator_container: Optional separate container for Navigator (minimap).
                                 If None, Navigator is created inside container.
        """
        self.container = container
        self.view_width = fixed_width
        self.view_height = fixed_height
        
        # Minimap settings (1/3 of main view)
        self.minimap_scale = 3
        self.minimap_width = fixed_width // self.minimap_scale
        self.minimap_height = fixed_height // self.minimap_scale
        self.minimap_dragging = False
        
        with container:
            # Viewer: Main image with scrollbars (fixed size container)
            with ui.column().classes('gap-0 flex-shrink-0'):
                # Row: image + vertical scrollbar
                with ui.row().classes('gap-1 items-stretch'):
                    # Image container with fixed size and overflow hidden
                    with ui.element('div').classes('relative flex-none') \
                            .style(f'width: {fixed_width}px; height: {fixed_height}px; overflow: hidden;') \
                            as self.scroll_container:
                        # Inner container for transform
                        with ui.element('div').classes('inline-block') \
                                .style('transform-origin: 0 0;') as self.transform_container:
                            self.image_component = ui.interactive_image(
                                source='',
                                on_mouse=self._handle_mouse,
                                events=['mousedown', 'mouseup', 'mousemove'],
                                cross=False,
                                sanitize=False,
                            ).classes('block')
                        
                        # Clean Annotations status label (positioned at top-left of image)
                        self.clean_status_label = ui.label('').classes('absolute text-sm font-bold').style(
                            'top: 10px; left: 10px; background-color: rgba(0,0,0,0.5); color: red; padding: 4px 8px; border-radius: 4px; display: none;'
                        )
                    
                    # Vertical scrollbar on right
                    self.v_scrollbar = ui.slider(
                        min=0, max=100, value=0,
                        on_change=self._on_v_scroll
                    ).props('vertical dense').style(f'height: {fixed_height}px;')
                
                # Horizontal scrollbar at bottom
                with ui.row().classes('gap-1'):
                    self.h_scrollbar = ui.slider(
                        min=0, max=100, value=0,
                        on_change=self._on_h_scroll
                    ).props('dense').style(f'width: {fixed_width}px;')
            
            # Set up keyboard listener
            self.keyboard = ui.keyboard(on_key=self._handle_key, ignore=['input', 'select', 'textarea'])
        
        # Navigator: Create in separate container if provided, otherwise skip
        # (Navigator can be created later via create_navigator method)
        if navigator_container is not None:
            self.create_navigator(navigator_container)
    
    def create_navigator(self, container) -> None:
        """Create Navigator (minimap) in the specified container
        
        Args:
            container: Container for the Navigator
        """
        with container:
            ui.label('Navigator').classes('text-xs text-gray-500')
            # Use a fixed container, same approach as Viewer
            # Fixed size container with overflow hidden
            with ui.element('div').classes('relative flex-none') \
                    .style(f'width: {self.minimap_width}px; height: {self.minimap_height}px; overflow: hidden; border: 1px solid #d1d5db; border-radius: 4px; cursor: pointer;') \
                    as self.minimap_container:
                # Inner container for the image (same structure as Viewer)
                with ui.element('div').classes('inline-block') \
                        .style('transform-origin: 0 0;') as self.minimap_transform_container:
                    self.minimap_component = ui.interactive_image(
                        source='',
                        on_mouse=self._handle_minimap_mouse,
                        events=['mousedown', 'mouseup', 'mousemove', 'click'],
                        cross=False,
                        sanitize=False,
                    ).classes('block')
        
        # Load current image into minimap if already loaded
        if self.image_path and self.image_width > 0 and self.image_height > 0:
            self.minimap_component.set_source(self.image_path)
            # Set explicit dimensions for the minimap image element
            self.minimap_component.style(f'width: {self.image_width}px; height: {self.image_height}px; display: block;')
            self._update_minimap()
    
    def load_image(self, image_path: str) -> None:
        """Load image and get its dimensions"""
        from pathlib import Path
        self.image_path = image_path
        path = Path(image_path)
        
        if path.exists():
            self.image_width, self.image_height = get_image_size(path)
            if self.image_component:
                self.image_component.set_source(image_path)
                # Set explicit dimensions for the image element so transform works correctly
                self.image_component.style(f'width: {self.image_width}px; height: {self.image_height}px; display: block;')
            # Also load into minimap
            if hasattr(self, 'minimap_component') and self.minimap_component:
                self.minimap_component.set_source(image_path)
                # Set explicit dimensions for the minimap image element
                self.minimap_component.style(f'width: {self.image_width}px; height: {self.image_height}px; display: block;')
            # Reset zoom when loading new image
            self.reset_zoom()
            # Reset current class to 0 for new boxes
            self.current_class = 0
        else:
            logger.warning(f"Image not found: {image_path}")
            self.image_width = 0
            self.image_height = 0
    
    def load_boxes(self, gt_boxes: List[BBox], pred_boxes: List[BBox]) -> None:
        """Load annotation boxes"""
        self.gt_boxes = deepcopy(gt_boxes)
        self.pred_boxes = deepcopy(pred_boxes)
        self.selected_box_id = None
        
        # Clear history and save initial state
        self.history.clear()
        self.history_index = -1
        self._save_history()
        
        self._update_display()
    
    def get_gt_boxes(self) -> List[BBox]:
        """Get current GT boxes"""
        return deepcopy(self.gt_boxes)
    
    def set_display_options(self, show_gt: bool, show_pred: bool) -> None:
        """Set display options"""
        self.show_gt = show_gt
        self.show_pred = show_pred
        self._update_display()
    
    def set_current_class(self, class_id: int) -> None:
        """Set default class for new boxes"""
        self.current_class = class_id
    
    def undo(self) -> bool:
        """Undo last action, returns success status"""
        if self.history_index > 0:
            self.history_index -= 1
            state = self.history[self.history_index]
            self.gt_boxes = deepcopy(state.gt_boxes)
            self.pred_boxes = deepcopy(state.pred_boxes)
            self.selected_box_id = state.selected_id
            self._update_display()
            self._notify_change()
            return True
        return False
    
    def redo(self) -> bool:
        """Redo undone action, returns success status"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.gt_boxes = deepcopy(state.gt_boxes)
            self.pred_boxes = deepcopy(state.pred_boxes)
            self.selected_box_id = state.selected_id
            self._update_display()
            self._notify_change()
            return True
        return False
    
    def get_selected_box(self) -> Optional[BBox]:
        """Get currently selected box (must be editable)"""
        if self.selected_box_id:
            # Check GT boxes
            for box in self.gt_boxes:
                if box.id == self.selected_box_id:
                    if getattr(box, 'editable', True):
                        return box
                    else:
                        # Non-editable box cannot be selected
                        self.selected_box_id = None
                        return None
            # Check Pred boxes
            for box in self.pred_boxes:
                if box.id == self.selected_box_id:
                    if getattr(box, 'editable', True):
                        return box
                    else:
                        self.selected_box_id = None
                        return None
        return None
    
    def get_all_boxes(self) -> List[BBox]:
        """Get all boxes (GT + Pred) for box list panel"""
        return self.gt_boxes + self.pred_boxes
    
    def set_box_visible(self, box_id: str, visible: bool) -> None:
        """Set visibility for a specific box"""
        for box in self.gt_boxes:
            if box.id == box_id:
                box.visible = visible
                self._update_display()
                return
        for box in self.pred_boxes:
            if box.id == box_id:
                box.visible = visible
                self._update_display()
                return
    
    def select_box_by_id(self, box_id: str) -> bool:
        """Select a box by ID (only if editable)"""
        # Check GT boxes
        for box in self.gt_boxes:
            if box.id == box_id:
                if getattr(box, 'editable', True):
                    self.selected_box_id = box_id
                    self._update_display()
                    self._notify_change()
                    return True
                return False
        # Check Pred boxes
        for box in self.pred_boxes:
            if box.id == box_id:
                if getattr(box, 'editable', True):
                    self.selected_box_id = box_id
                    self._update_display()
                    self._notify_change()
                    return True
                return False
        return False
    
    def _zoom_to_box(self, box: BBox) -> None:
        """Zoom to center on the specified box"""
        # Calculate box center in image coordinates
        box_center_x = box.x + box.w / 2
        box_center_y = box.y + box.h / 2
        
        # Zoom to level 3x (or appropriate level)
        target_zoom = 3.0
        self.set_zoom(target_zoom, focus_point=(box_center_x, box_center_y))
    
    # ==================== One-Click Actions ====================
    
    def swap_editable(self) -> None:
        """Swap editable status of all boxes
        
        All editable=True boxes become editable=False
        All editable=False boxes become editable=True
        """
        for box in self.gt_boxes:
            box.editable = not getattr(box, 'editable', True)
        for box in self.pred_boxes:
            box.editable = not getattr(box, 'editable', True)
        
        # Clear selection (selected box may no longer be editable)
        self.selected_box_id = None
        
        self._save_history()
        self._notify_change()
        self._update_display()
    
    def clear_editable(self) -> None:
        """Delete all editable boxes"""
        # Remove all editable boxes
        self.gt_boxes = [b for b in self.gt_boxes if not getattr(b, 'editable', True)]
        self.pred_boxes = [b for b in self.pred_boxes if not getattr(b, 'editable', True)]
        
        # Clear selection
        self.selected_box_id = None
        
        self._save_history()
        self._notify_change()
        self._update_display()
    
    def activate_reference(self) -> None:
        """Make all non-editable boxes editable"""
        for box in self.gt_boxes:
            if not getattr(box, 'editable', True):
                box.editable = True
        for box in self.pred_boxes:
            if not getattr(box, 'editable', True):
                box.editable = True
        
        self._save_history()
        self._notify_change()
        self._update_display()
    
    # ==================== Zoom Methods ====================
    
    def _get_max_pan(self) -> tuple:
        """Get maximum pan values based on current zoom
        
        Returns:
            (max_pan_x, max_pan_y) in image coordinates
        """
        if self.zoom <= 1.0:
            return (0, 0)
        scale_1x = self._get_viewer_scale_1x()
        visible_width = self.view_width / (self.zoom * scale_1x)
        visible_height = self.view_height / (self.zoom * scale_1x)
        max_pan_x = max(0, self.image_width - visible_width)
        max_pan_y = max(0, self.image_height - visible_height)
        return (max_pan_x, max_pan_y)
    
    def _on_h_scroll(self, e) -> None:
        """Handle horizontal scrollbar change"""
        if self.zoom <= 1.0 or self.image_width == 0:
            return
        
        # Convert scrollbar value (0-100) to pan position
        max_pan_x, _ = self._get_max_pan()
        if max_pan_x > 0:
            self.pan_x = (e.value / 100) * max_pan_x
            self._apply_transform()
    
    def _on_v_scroll(self, e) -> None:
        """Handle vertical scrollbar change"""
        if self.zoom <= 1.0 or self.image_height == 0:
            return
        
        # Convert scrollbar value (0-100) to pan position
        _, max_pan_y = self._get_max_pan()
        if max_pan_y > 0:
            self.pan_y = (e.value / 100) * max_pan_y
            self._apply_transform()
    
    def _update_scrollbars(self) -> None:
        """Update scrollbar positions based on current pan"""
        if not hasattr(self, 'h_scrollbar') or not hasattr(self, 'v_scrollbar'):
            return
        
        if self.zoom <= 1.0:
            self.h_scrollbar.set_value(0)
            self.v_scrollbar.set_value(0)
            self.h_scrollbar.set_visibility(False)
            self.v_scrollbar.set_visibility(False)
        else:
            max_pan_x, max_pan_y = self._get_max_pan()

            self.h_scrollbar.set_visibility(max_pan_x > 0)
            self.v_scrollbar.set_visibility(max_pan_y > 0)
            
            if max_pan_x > 0:
                h_value = (self.pan_x / max_pan_x) * 100
                self.h_scrollbar.set_value(h_value)
            else:
                self.h_scrollbar.set_value(0)
            
            if max_pan_y > 0:
                v_value = (self.pan_y / max_pan_y) * 100
                self.v_scrollbar.set_value(v_value)
            else:
                self.v_scrollbar.set_value(0)
    
    def set_zoom(self, zoom: float, focus_point: tuple = None) -> None:
        """Set zoom level (1.0 = 100%)
        
        Args:
            zoom: Target zoom level
            focus_point: (x, y) in image coordinates to keep centered. If None, auto-determine.
        """
        # Clamp to range (1.0 - 20.0)
        new_zoom = max(1.0, min(zoom, 20.0))
        
        if new_zoom == self.zoom:
            return
        
        # Determine focus point (in image coordinates)
        if focus_point is None:
            focus_point = self._get_zoom_focus_point()
        
        focus_x, focus_y = focus_point
        
        # Update zoom
        self.zoom = new_zoom
        
        if self.zoom == 1.0:
            # At 1x zoom, no panning
            self.pan_x = 0
            self.pan_y = 0
        else:
            # Calculate new pan to keep focus point at the same screen position
            # At zoom>1, the visible area in image coordinates is:
            #   width = view_width / (zoom * scale_1x)
            #   height = view_height / (zoom * scale_1x)
            # We want focus_point to be at the center of the visible area
            scale_1x = self._get_viewer_scale_1x()
            visible_w = self.view_width / (self.zoom * scale_1x)
            visible_h = self.view_height / (self.zoom * scale_1x)
            
            self.pan_x = focus_x - visible_w / 2
            self.pan_y = focus_y - visible_h / 2
        
        self._constrain_pan()
        self._apply_transform()
        self._update_scrollbars()
        
        if self.on_zoom_change:
            self.on_zoom_change(self.zoom)
    
    def _get_zoom_focus_point(self) -> tuple:
        """Get the focus point for zooming (in image coordinates)
        
        Priority:
        1. If a box is selected, focus on the selected box center
        2. Otherwise, focus on the center of the current view
        
        Returns:
            (x, y) - focus point in image coordinates
        """
        # Check if a box is selected
        selected_box = self.get_selected_box()
        if selected_box:
            # Focus on selected box center
            cx = selected_box.x + selected_box.w / 2
            cy = selected_box.y + selected_box.h / 2
            return (cx, cy)
        
        # No selection - focus on current view center
        # Calculate the center of the currently visible area in image coordinates
        scale_1x = self._get_viewer_scale_1x()
        
        if self.zoom > 1.0:
            # View center in image coordinates when zoomed
            visible_w = self.view_width / (self.zoom * scale_1x)
            visible_h = self.view_height / (self.zoom * scale_1x)
            cx = self.pan_x + visible_w / 2
            cy = self.pan_y + visible_h / 2
        else:
            # At 1x zoom, calculate the center of the visible area
            # The image is displayed with scale_1x, and may be centered in the viewer
            # We need to find what part of the image is visible at the center of the viewer
            # Viewer center in display coordinates: (view_width/2, view_height/2)
            # Convert to image coordinates
            display_width_1x = self.image_width * scale_1x
            display_height_1x = self.image_height * scale_1x
            display_x_1x = (self.view_width - display_width_1x) / 2
            display_y_1x = (self.view_height - display_height_1x) / 2
            
            # Viewer center in display coordinates
            viewer_center_x_display = self.view_width / 2
            viewer_center_y_display = self.view_height / 2
            
            # Convert viewer center from display coordinates to image coordinates
            # First, subtract the display offset to get position relative to scaled image
            relative_x = viewer_center_x_display - display_x_1x
            relative_y = viewer_center_y_display - display_y_1x
            
            # Then convert from scaled display coordinates to image coordinates
            cx = relative_x / scale_1x
            cy = relative_y / scale_1x
            
            # Clamp to image bounds
            cx = max(0, min(cx, self.image_width))
            cy = max(0, min(cy, self.image_height))
        
        return (cx, cy)
    
    def get_zoom(self) -> float:
        """Get current zoom level"""
        return self.zoom
    
    def reset_zoom(self) -> None:
        """Reset zoom to 1x and clear pan"""
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._apply_transform()
        self._update_scrollbars()
        
        if self.on_zoom_change:
            self.on_zoom_change(self.zoom)
    
    def get_view_state(self) -> dict:
        """Get current view state (zoom and pan)"""
        return {
            'zoom': self.zoom,
            'pan_x': self.pan_x,
            'pan_y': self.pan_y
        }
    
    def set_view_state(self, view_state: dict) -> None:
        """Set view state (zoom and pan)"""
        if 'zoom' in view_state:
            self.zoom = max(1.0, min(float(view_state['zoom']), 20.0))
        if 'pan_x' in view_state:
            self.pan_x = float(view_state['pan_x'])
        if 'pan_y' in view_state:
            self.pan_y = float(view_state['pan_y'])

        self._constrain_pan()

        # Apply transform and update scrollbars
        self._apply_transform()
        self._update_scrollbars()
        
        if self.on_zoom_change:
            self.on_zoom_change(self.zoom)
    
    def zoom_in(self) -> None:
        """Zoom in by small step"""
        self.set_zoom(self.zoom + self.ZOOM_STEP)
    
    def zoom_out(self) -> None:
        """Zoom out by small step"""
        self.set_zoom(self.zoom - self.ZOOM_STEP)
    
    def auto_focus_boxes(self) -> None:
        """Automatically set zoom and pan to focus on all boxes (GT + Pred)
        
        Calculates the union of all boxes and sets the optimal zoom level and pan
        position to show all boxes in the viewport.
        """
        # Collect all boxes
        all_boxes = self.gt_boxes + self.pred_boxes
        if not all_boxes:
            return
        
        # Calculate union of all boxes
        # Use all boxes' coordinates - no box should be left out
        min_x = min(box.x for box in all_boxes)
        min_y = min(box.y for box in all_boxes)
        max_x = max(box.x + box.w for box in all_boxes)
        max_y = max(box.y + box.h for box in all_boxes)
        
        # Calculate symmetric buffer that considers the entire viewport
        # Use the same buffer for all sides
        buffer = 40  # Balanced buffer to ensure all boxes and labels are visible
        
        # Apply symmetric buffer
        min_x -= buffer
        min_y -= buffer
        max_x += buffer
        max_y += buffer
        
        # Ensure boundaries don't go below 0
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        
        # Box dimensions
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        if box_width <= 0 or box_height <= 0:
            return
        
        # Get scale at 1x zoom
        scale_1x = self._get_viewer_scale_1x()
        if scale_1x <= 0:
            return
        
        # Calculate optimal zoom level with appropriate margin
        # At zoom level Z, visible area in image coords: view_width/(Z*scale_1x) x view_height/(Z*scale_1x)
        # We want: visible_width >= box_width and visible_height >= box_height with sufficient margin
        margin_factor = 0.85  # 15% margin to ensure all boxes and labels fit comfortably
        
        # Calculate required zoom for x and y directions
        visible_width_needed = box_width / margin_factor
        visible_height_needed = box_height / margin_factor
        
        zoom_x = self.view_width / (visible_width_needed * scale_1x)
        zoom_y = self.view_height / (visible_height_needed * scale_1x)
        
        optimal_zoom = min(zoom_x, zoom_y)
        
        # Clamp to range (1.0 - 20.0)
        optimal_zoom = max(1.0, min(optimal_zoom, 20.0))
        
        # Calculate center of boxes - use the original box center without buffer
        # This ensures we're focusing on the actual content
        original_center_x = (min(box.x for box in all_boxes) + max(box.x + box.w for box in all_boxes)) / 2
        original_center_y = (min(box.y for box in all_boxes) + max(box.y + box.h for box in all_boxes)) / 2
        
        # Set zoom using the set_zoom method with the original center of boxes as focus point
        self.set_zoom(float(optimal_zoom), focus_point=(original_center_x, original_center_y))
        
        # The set_zoom method handles pan calculation, constraint, transform application,
        # scrollbar updates, and zoom change notifications
    
    def _get_viewer_scale_1x(self) -> float:
        """Get the scale factor at 1x zoom (when image fits in viewer with aspect ratio preserved)
        
        Returns:
            scale factor (display_width / image_width, should equal display_height / image_height)
        """
        if self.image_width <= 0 or self.image_height <= 0:
            return 1.0
        
        scale_x = self.view_width / self.image_width
        scale_y = self.view_height / self.image_height
        return min(scale_x, scale_y)  # Use smaller scale to fit both dimensions
    
    def _apply_transform(self) -> None:
        """Apply CSS transform for zoom and pan
        
        The key insight: pan_x and pan_y are in IMAGE coordinates (not display coordinates).
        pan_x and pan_y represent the top-left corner of the visible viewport in image coordinates.
        At 1x zoom, the image is displayed with scale_1x to fit the container and centered.
        At zoom>1, we scale by zoom, and translate to show the region starting from (pan_x, pan_y).
        
        CSS transforms are applied right-to-left, so:
        transform: translate(x, y) scale(z) means: first scale, then translate
        
        Strategy:
        1. Always apply transform to scale and center the image (even at 1x zoom)
        2. At 1x zoom: scale by scale_1x and center the image
        3. At zoom>1: scale by zoom*scale_1x, and translate to show the region starting from (pan_x, pan_y)
        """
        if not hasattr(self, 'transform_container') or not self.transform_container:
            return
        
        if self.image_width <= 0 or self.image_height <= 0:
            return
        
        # Get scale at 1x zoom
        scale_1x = self._get_viewer_scale_1x()
        
        # Calculate display offset at 1x (for centering when aspect ratio differs)
        display_width_1x = self.image_width * scale_1x
        display_height_1x = self.image_height * scale_1x
        display_x_1x = (self.view_width - display_width_1x) / 2
        display_y_1x = (self.view_height - display_height_1x) / 2
        
        if self.zoom <= 1.0:
            # At 1x zoom, scale and center the image
            transform = f'translate({display_x_1x}px, {display_y_1x}px) scale({scale_1x})'
            self.transform_container.style(f'transform: {transform}; transform-origin: 0 0;')
        else:
            scaled_width = self.image_width * scale_1x * self.zoom
            scaled_height = self.image_height * scale_1x * self.zoom

            # Keep content centered on any axis where the zoomed image is still
            # smaller than the viewport; otherwise pan from the top-left edge.
            base_translate_x = max(0.0, (self.view_width - scaled_width) / 2)
            base_translate_y = max(0.0, (self.view_height - scaled_height) / 2)

            translate_x = base_translate_x - (self.pan_x * scale_1x * self.zoom)
            translate_y = base_translate_y - (self.pan_y * scale_1x * self.zoom)

            transform = f'translate({translate_x}px, {translate_y}px) scale({self.zoom * scale_1x})'
            self.transform_container.style(f'transform: {transform}; transform-origin: 0 0;')
        
        # Update minimap viewport indicator
        self._update_minimap()
    
    def _get_minimap_display_info(self) -> Tuple[float, float, float, float]:
        """Calculate the actual displayed image area in Navigator container
        
        Navigator uses the same approach as Viewer: fixed container, image scales naturally
        to fill the longer dimension, with padding on the shorter dimension.
        
        Returns:
            (display_x, display_y, display_width, display_height) - position and size of displayed image
            in Navigator container coordinates (pixels)
        """
        if self.image_width <= 0 or self.image_height <= 0:
            return (0, 0, self.minimap_width, self.minimap_height)
        
        # Calculate scale to fit image in container while maintaining aspect ratio
        # Same logic as Viewer: scale so that the longer dimension fills the container
        scale_x = self.minimap_width / self.image_width
        scale_y = self.minimap_height / self.image_height
        scale = min(scale_x, scale_y)  # Use smaller scale to fit both dimensions
        
        # Calculate actual displayed size (same as Viewer)
        display_width = self.image_width * scale
        display_height = self.image_height * scale
        
        # Center the image in the container (same as Viewer)
        display_x = (self.minimap_width - display_width) / 2
        display_y = (self.minimap_height - display_height) / 2
        
        return (display_x, display_y, display_width, display_height)
    
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
    
    def _handle_minimap_mouse(self, e) -> None:
        """Handle mouse events on minimap image
        
        Note: interactive_image returns coordinates relative to the ORIGINAL image size,
        not the displayed size. With object-fit: contain, we need to account for the
        actual displayed area and scale.
        """
        if not hasattr(e, 'image_x') or not hasattr(e, 'image_y'):
            return
        
        # interactive_image gives us coordinates in the original image coordinate system
        # These coordinates are already in image pixel coordinates, which is what we need
        image_x = e.image_x
        image_y = e.image_y
        
        # Clamp to image bounds (interactive_image should already do this, but be safe)
        image_x = max(0, min(image_x, self.image_width))
        image_y = max(0, min(image_y, self.image_height))
        
        event_type = e.type if hasattr(e, 'type') else 'unknown'
        
        if event_type == 'click' or event_type == 'mousedown':
            self.minimap_dragging = (event_type == 'mousedown')
            # When clicking on minimap, center the viewport on the clicked point
            self._minimap_set_center(image_x, image_y)
        elif event_type == 'mousemove' and self.minimap_dragging:
            # When dragging, update the viewport center
            self._minimap_set_center(image_x, image_y)
        elif event_type == 'mouseup':
            self.minimap_dragging = False
    
    
    def _minimap_set_center(self, image_x: float, image_y: float) -> None:
        """Set the main view pan so that the viewport is centered at the given image position
        
        Args:
            image_x, image_y: Coordinates in image coordinate system
        """
        if self.zoom <= 1.0:
            return
        
        # Set pan so that (image_x, image_y) is at the center of the viewport
        scale_1x = self._get_viewer_scale_1x()
        visible_w = self.view_width / (self.zoom * scale_1x)
        visible_h = self.view_height / (self.zoom * scale_1x)
        
        self.pan_x = image_x - visible_w / 2
        self.pan_y = image_y - visible_h / 2
        
        self._constrain_pan()
        self._apply_transform()
        self._update_scrollbars()
    
    def _constrain_pan(self) -> None:
        """Constrain pan to strict image boundaries."""
        if self.zoom <= 1.0:
            self.pan_x = 0
            self.pan_y = 0
            return
        
        # Pan is in image coordinates
        # At zoom Z with scale_1x, visible area in image coords is (view_width/(Z*scale_1x), view_height/(Z*scale_1x))
        # Maximum pan is image_size - visible_area
        scale_1x = self._get_viewer_scale_1x()
        visible_width = self.view_width / (self.zoom * scale_1x)
        visible_height = self.view_height / (self.zoom * scale_1x)

        min_pan_x = 0.0
        min_pan_y = 0.0
        max_pan_x = max(0.0, self.image_width - visible_width)
        max_pan_y = max(0.0, self.image_height - visible_height)
        
        self.pan_x = max(min_pan_x, min(self.pan_x, max_pan_x))
        self.pan_y = max(min_pan_y, min(self.pan_y, max_pan_y))
    
    # ==================== Private Methods ====================
    
    def _save_history(self) -> None:
        """Save current state to history (includes both gt_boxes and pred_boxes)"""
        # Remove any redo states
        self.history = self.history[:self.history_index + 1]
        
        # Add new state (save both gt and pred boxes with all attributes)
        state = HistoryState(
            gt_boxes=deepcopy(self.gt_boxes),
            pred_boxes=deepcopy(self.pred_boxes),
            selected_id=self.selected_box_id
        )
        self.history.append(state)
        
        # Trim old history
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]
        
        self.history_index = len(self.history) - 1
    
    def _notify_change(self) -> None:
        """Notify about GT boxes change"""
        if self.on_change:
            self.on_change(self.get_gt_boxes())
    
    def _update_display(self) -> None:
        """Update SVG overlay content
        
        Only renders boxes where:
        - Global show flag is True (show_gt or show_pred)
        - Individual box visible attribute is True
        
        Implements intelligent label positioning:
        - GT labels on top-left, Pred labels on top-right
        - No overlapping labels through vertical offset
        - Merged multi-line labels for highly overlapping cases
        - Boundary-friendly positioning
        """
        if not self.image_component:
            return
        
        svg_parts = []
        
        # Track rendered labels for each source type (GT/Pred)
        # Structure: {source: [{label_info}, ...]}
        # label_info: {x, y, width, height, box_ids, labels, colors}
        rendered_labels = {
            'gt': [],
            'pred': []
        }
        
        # First render all boxes (without labels)
        # Render GT boxes
        if self.show_gt:
            for box in self.gt_boxes:
                if getattr(box, 'visible', True):
                    svg_parts.append(self._render_box_without_label(box))
        
        # Render Pred boxes
        if self.show_pred:
            for box in self.pred_boxes:
                if getattr(box, 'visible', True):
                    svg_parts.append(self._render_box_without_label(box))
        
        # First pass: collect all labels and detect overlaps for merging
        if self.show_gt:
            for box in self.gt_boxes:
                if getattr(box, 'visible', True):
                    self._collect_label_info(box, rendered_labels)
        
        if self.show_pred:
            for box in self.pred_boxes:
                if getattr(box, 'visible', True):
                    self._collect_label_info(box, rendered_labels)
        
        # Second pass: render all collected labels (including merged ones)
        self._render_all_labels(rendered_labels, svg_parts)
        
        # Render handles for selected box (only if editable and visible)
        selected_box = self.get_selected_box()
        if selected_box and self.show_gt:
            if getattr(selected_box, 'visible', True) and getattr(selected_box, 'editable', True):
                svg_parts.append(self._render_handles(selected_box))
        
        self.image_component.set_content(''.join(svg_parts))
    
    def _collect_label_info(self, box: BBox, rendered_labels: dict) -> None:
        """Collect label information
        
        Args:
            box: Bounding box to process
            rendered_labels: Dictionary to track all rendered labels
        """
        source = box.source.value if hasattr(box.source, 'value') else box.source
        
        # Label text
        label = f"{box.class_id}"
        
        # Calculate background width based on number of digits
        bg_width = 20 if box.class_id < 10 else 28
        label_height = 18
        
        # Check if box is selected
        is_selected = box.id == self.selected_box_id
        color = self.COLORS[source]['selected' if is_selected else 'normal']
        
        # Calculate possible label positions around the box
        # Order: top-left, top-right, bottom-left, bottom-right
        possible_positions = [
            # Top-left
            {'x': box.x, 'y': box.y - label_height},
            # Top-right
            {'x': box.x + box.w - bg_width, 'y': box.y - label_height},
            # Bottom-left
            {'x': box.x, 'y': box.y + box.h},
            # Bottom-right
            {'x': box.x + box.w - bg_width, 'y': box.y + box.h}
        ]
        
        # Filter positions to ensure they stay within image boundaries
        valid_positions = []
        for pos in possible_positions:
            if (pos['x'] >= 0 and 
                pos['y'] >= 0 and 
                pos['x'] + bg_width <= self.image_width and 
                pos['y'] + label_height <= self.image_height):
                valid_positions.append(pos)
        
        # Use the first valid position if available
        if valid_positions:
            base_pos = valid_positions[0]
        else:
            # Fallback to top-left if no valid positions
            base_pos = {'x': box.x, 'y': box.y - label_height}
        
        # No merging, always track as new label
        rendered_labels[source].append({
            'x': base_pos['x'],
            'y': base_pos['y'],
            'width': bg_width,
            'height': label_height,
            'box_ids': [box.id],
            'labels': [label],
            'colors': [color]
        })
    
    def _render_all_labels(self, rendered_labels: dict, svg_parts: list) -> None:
        """Render all collected labels with intelligent positioning
        
        Args:
            rendered_labels: Dictionary of all labels to render
            svg_parts: List to append SVG content to
        """
        for source, labels in rendered_labels.items():
            # For each source, find non-overlapping positions for all labels
            positioned_labels = []
            
            for label_info in labels:
                # Find non-overlapping position for this label
                x, y = self._find_non_overlapping_position(label_info, positioned_labels)
                
                # Ensure label stays within image boundaries
                label_height = 18
                max_width = max(20 if int(label) < 10 else 28 for label in label_info['labels'])
                
                x = max(0, x)
                y = max(0, y)
                
                # Check if label would go beyond image boundaries
                if x + max_width > self.image_width:
                    x = self.image_width - max_width
                if y + label_info['height'] > self.image_height:
                    y = self.image_height - label_info['height']
                
                # Update label position
                positioned_label = label_info.copy()
                positioned_label['x'] = x
                positioned_label['y'] = y
                positioned_label['width'] = max_width
                positioned_labels.append(positioned_label)
            
            # Now render all positioned labels
            for label_info in positioned_labels:
                # Render background rectangle
                svg_parts.append(f'''<rect x="{label_info['x']}" y="{label_info['y']}" width="{label_info['width']}" height="{label_info['height']}" 
                      fill="white" fill-opacity="0.85" rx="2"/>''')
                
                # Render each label in the multi-line label
                for i, (label, color) in enumerate(zip(label_info['labels'], label_info['colors'])):
                    text_y = label_info['y'] + 15 + (i * 18)  # 15 is for vertical centering in each line
                    svg_parts.append(f'''<text x="{label_info['x'] + 3}" y="{text_y}" 
                        font-size="16" font-family="Arial" font-weight="bold"
                        fill="{color}">{label}</text>''')
    
    def _find_non_overlapping_position(self, new_label: dict, positioned_labels: list) -> tuple:
        """Find a non-overlapping position for a new label
        
        Args:
            new_label: New label to position
            positioned_labels: List of already positioned labels
        
        Returns:
            (x, y) - Position coordinates for the new label
        """
        width = new_label['width']
        height = new_label['height']
        
        # Get the box ID to find its coordinates
        box_id = new_label['box_ids'][0]
        # Find the corresponding box
        box = None
        for b in self.gt_boxes + self.pred_boxes:
            if b.id == box_id:
                box = b
                break
        
        if not box:
            # Fallback to original position if box not found
            return (new_label['x'], new_label['y'])
        
        # Calculate all possible positions around the box
        possible_positions = [
            # Top-left
            (box.x, box.y - height),
            # Top-right
            (box.x + box.w - width, box.y - height),
            # Bottom-left
            (box.x, box.y + box.h),
            # Bottom-right
            (box.x + box.w - width, box.y + box.h),
            # Left-center
            (box.x - width, box.y + (box.h - height) / 2),
            # Right-center
            (box.x + box.w, box.y + (box.h - height) / 2)
        ]
        
        # Check each position for overlap and boundaries
        for pos_x, pos_y in possible_positions:
            # Check if position is within image boundaries
            if (pos_x >= 0 and 
                pos_y >= 0 and 
                pos_x + width <= self.image_width and 
                pos_y + height <= self.image_height):
                
                # Check if position overlaps with any existing label
                overlap_found = False
                for existing_label in positioned_labels:
                    overlap_x1 = max(pos_x, existing_label['x'])
                    overlap_y1 = max(pos_y, existing_label['y'])
                    overlap_x2 = min(pos_x + width, existing_label['x'] + existing_label['width'])
                    overlap_y2 = min(pos_y + height, existing_label['y'] + existing_label['height'])
                    
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_found = True
                        break
                
                if not overlap_found:
                    return (pos_x, pos_y)
        
        # If no good position found, use the original position
        return (new_label['x'], new_label['y'])
    
    def _render_box(self, box: BBox) -> str:
        """Render a single bounding box as SVG
        
        - editable=True: solid line
        - editable=False: dashed line
        - Color is determined by source (GT=green, Pred=blue)
        """
        is_selected = box.id == self.selected_box_id
        source = box.source.value if isinstance(box.source, BoxSource) else box.source
        
        color = self.COLORS[source]['selected' if is_selected else 'normal']
        stroke_width = 3 if is_selected else 2
        
        # Editable boxes use solid line, non-editable use dashed line
        editable = getattr(box, 'editable', True)
        dash_array = '' if editable else '5,5'
        
        # Box rectangle
        rect_attrs = f'x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}"'
        rect_style = f'fill="none" stroke="{color}" stroke-width="{stroke_width}"'
        if dash_array:
            rect_style += f' stroke-dasharray="{dash_array}"'
        
        svg = f'<rect {rect_attrs} {rect_style} data-box-id="{box.id}"/>'
        
        # Label text
        label = f"{box.class_id}"
        
        # Label position (above the box, or inside if too close to top)
        label_x = box.x
        label_y = box.y - 6
        if label_y < 18:
            label_y = box.y + 18
        
        # Calculate background width based on number of digits
        bg_width = 20 if box.class_id < 10 else 28
        
        # Add background rectangle for better readability
        svg += f'''<rect x="{label_x}" y="{label_y - 14}" width="{bg_width}" height="18" 
              fill="white" fill-opacity="0.85" rx="2"/>'''
        
        # Label text with larger font
        svg += f'''<text x="{label_x + 3}" y="{label_y}" 
            font-size="16" font-family="Arial" font-weight="bold"
            fill="{color}">{label}</text>'''
        
        return svg
    
    def _render_box_without_label(self, box: BBox) -> str:
        """Render a single bounding box as SVG without label
        
        - editable=True: solid line
        - editable=False: dashed line
        - Color is determined by source (GT=green, Pred=blue)
        - Selected boxes are red
        """
        source = box.source.value if hasattr(box.source, 'value') else box.source
        
        # Check if box is selected
        is_selected = box.id == self.selected_box_id
        color = self.COLORS[source]['selected'] if is_selected else self.COLORS[source]['normal']
        stroke_width = 2
        
        # Editable boxes use solid line, non-editable use dashed line
        editable = getattr(box, 'editable', True)
        dash_array = '' if editable else '5,5'
        
        # Box rectangle
        rect_attrs = f'x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}"'
        rect_style = f'fill="none" stroke="{color}" stroke-width="{stroke_width}"'
        if dash_array:
            rect_style += f' stroke-dasharray="{dash_array}"'
        
        return f'<rect {rect_attrs} {rect_style} data-box-id="{box.id}"/>'
    

    
    def _render_handles(self, box: BBox) -> str:
        """Render resize handles for selected box"""
        x, y, w, h = box.x, box.y, box.w, box.h
        # Use smaller handle size for better visibility
        handle_size = 1.8
        
        positions = {
            'nw': (x, y),
            'n':  (x + w/2, y),
            'ne': (x + w, y),
            'e':  (x + w, y + h/2),
            'se': (x + w, y + h),
            's':  (x + w/2, y + h),
            'sw': (x, y + h),
            'w':  (x, y + h/2),
        }
        
        cursors = {
            'nw': 'nw-resize', 'ne': 'ne-resize',
            'sw': 'sw-resize', 'se': 'se-resize',
            'n': 'n-resize', 's': 's-resize',
            'e': 'e-resize', 'w': 'w-resize',
        }
        
        svg_parts = []
        for pos_name, (cx, cy) in positions.items():
            svg_parts.append(
                f'<rect x="{cx - handle_size/2}" y="{cy - handle_size/2}" '
                f'width="{handle_size}" height="{handle_size}" '
                f'fill="#ef4444" stroke="#ef4444" stroke-width="1" '
                f'data-handle="{pos_name}" style="cursor: {cursors[pos_name]}"/>'
            )
        
        return ''.join(svg_parts)
    
    def _handle_mouse(self, e) -> None:
        """Handle mouse events"""
        if not self.image_width or not self.image_height:
            return
        
        event_type = e.type
        
        # Get coordinates from event
        # image_x/image_y are coordinates relative to the image
        x = e.image_x if hasattr(e, 'image_x') else 0
        y = e.image_y if hasattr(e, 'image_y') else 0
        
        if event_type == 'mousedown':
            self._on_mouse_down(x, y)
        elif event_type == 'mousemove':
            self._on_mouse_move(x, y)
        elif event_type == 'mouseup':
            self._on_mouse_up(x, y)
    
    def _on_mouse_down(self, x: float, y: float) -> None:
        """Handle mouse down event"""
        target = self._get_click_target(x, y)
        
        if target[0] == 'handle':
            # Start resize
            self.drag_mode = 'resize'
            self.drag_handle = target[1]
            self.drag_start_x = x
            self.drag_start_y = y
            selected_box = self.get_selected_box()
            if selected_box:
                self.drag_box_start = {
                    'x': selected_box.x, 'y': selected_box.y,
                    'w': selected_box.w, 'h': selected_box.h
                }
        elif target[0] == 'box':
            box = target[1]
            # Select the box
            self.selected_box_id = box.id
            # Start move
            self.drag_mode = 'move'
            self.drag_start_x = x
            self.drag_start_y = y
            self.drag_box_start = {
                'x': box.x, 'y': box.y, 'w': box.w, 'h': box.h
            }
            self._update_display()
        else:
            # Click on empty area
            if self.selected_box_id:
                # Deselect
                self.selected_box_id = None
                self._update_display()
            else:
                # Start creating new box (works at any zoom level)
                # interactive_image provides image_x/image_y in image coordinates, unaffected by CSS transform
                self.drag_mode = 'create'
                self.drag_start_x = x
                self.drag_start_y = y
                self.drag_box_start = None
    
    def _on_mouse_move(self, x: float, y: float) -> None:
        """Handle mouse move event"""
        if not self.drag_mode:
            return
        
        if self.drag_mode == 'move':
            self._do_move(x, y)
        elif self.drag_mode == 'resize':
            self._do_resize(x, y)
        elif self.drag_mode == 'create':
            self._do_create(x, y)
        elif self.drag_mode == 'pan':
            self._do_pan(x, y)
    
    def _on_mouse_up(self, x: float, y: float) -> None:
        """Handle mouse up event"""
        if self.drag_mode == 'create':
            self._finish_create(x, y)
        elif self.drag_mode in ('move', 'resize'):
            # Save history after edit
            self._save_history()
            self._notify_change()
        
        self.drag_mode = None
        self.drag_handle = None
        self.drag_box_start = None
    
    def _do_move(self, x: float, y: float) -> None:
        """Move selected box"""
        if not self.drag_box_start:
            return
        
        selected_box = self.get_selected_box()
        if not selected_box:
            return
        
        dx = x - self.drag_start_x
        dy = y - self.drag_start_y
        
        new_x = self.drag_box_start['x'] + dx
        new_y = self.drag_box_start['y'] + dy
        
        # Constrain to image bounds
        new_x = max(0, min(new_x, self.image_width - selected_box.w))
        new_y = max(0, min(new_y, self.image_height - selected_box.h))
        
        selected_box.x = new_x
        selected_box.y = new_y
        
        self._update_display()
    
    def _do_resize(self, x: float, y: float) -> None:
        """Resize selected box via handle drag"""
        if not self.drag_box_start or not self.drag_handle:
            return
        
        selected_box = self.get_selected_box()
        if not selected_box:
            return
        
        dx = x - self.drag_start_x
        dy = y - self.drag_start_y
        
        orig = self.drag_box_start
        new_x, new_y = orig['x'], orig['y']
        new_w, new_h = orig['w'], orig['h']
        
        handle = self.drag_handle
        
        # Adjust based on handle position
        if 'w' in handle:  # Left side
            new_x = orig['x'] + dx
            new_w = orig['w'] - dx
        if 'e' in handle:  # Right side
            new_w = orig['w'] + dx
        if 'n' in handle:  # Top side
            new_y = orig['y'] + dy
            new_h = orig['h'] - dy
        if 's' in handle:  # Bottom side
            new_h = orig['h'] + dy
        
        # Apply constraints
        new_x, new_y, new_w, new_h = self._constrain_box(new_x, new_y, new_w, new_h)
        
        selected_box.x = new_x
        selected_box.y = new_y
        selected_box.w = new_w
        selected_box.h = new_h
        
        self._update_display()
    
    def _do_pan(self, x: float, y: float) -> None:
        """Pan the view when dragging in zoomed mode"""
        # Calculate delta in image coordinates
        dx = x - self.drag_start_x
        dy = y - self.drag_start_y
        
        # Update pan (reverse direction for natural feel)
        self.pan_x = self.pan_start_x - dx
        self.pan_y = self.pan_start_y - dy
        
        # Constrain pan
        self._constrain_pan()
        
        # Apply transform and update scrollbars
        self._apply_transform()
        self._update_scrollbars()
    
    def _do_create(self, x: float, y: float) -> None:
        """Update box being created"""
        import uuid
        # Calculate box dimensions
        x1, y1 = self.drag_start_x, self.drag_start_y
        x2, y2 = x, y
        
        # Normalize coordinates
        box_x = min(x1, x2)
        box_y = min(y1, y2)
        box_w = abs(x2 - x1)
        box_h = abs(y2 - y1)
        
        if box_w < 2 or box_h < 2:
            return
        
        # Constrain to image
        box_x, box_y, box_w, box_h = self._constrain_box(box_x, box_y, box_w, box_h)
        
        # Update or create temporary box
        if not self.drag_box_start:
            # Create new temporary box
            new_box = BBox(
                x=box_x, y=box_y, w=box_w, h=box_h,
                class_id=self.current_class,
                source=BoxSource.GT,
                id=str(uuid.uuid4())
            )
            self.gt_boxes.append(new_box)
            self.selected_box_id = new_box.id
            self.drag_box_start = {'id': new_box.id}
        else:
            # Update existing temporary box
            for box in self.gt_boxes:
                if box.id == self.drag_box_start['id']:
                    box.x = box_x
                    box.y = box_y
                    box.w = box_w
                    box.h = box_h
                    break
        
        self._update_display()
    
    def _finish_create(self, x: float, y: float) -> None:
        """Finish creating a new box"""
        if not self.drag_box_start:
            return
        
        # Find the created box
        created_box = None
        for box in self.gt_boxes:
            if box.id == self.drag_box_start.get('id'):
                created_box = box
                break
        
        if created_box:
            # Check minimum size
            if created_box.w < self.MIN_BOX_SIZE or created_box.h < self.MIN_BOX_SIZE:
                # Remove too-small box
                self.gt_boxes = [b for b in self.gt_boxes if b.id != created_box.id]
                self.selected_box_id = None
            else:
                # Save history
                self._save_history()
                self._notify_change()
        
        self._update_display()
    
    def _constrain_box(self, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """Constrain box to image bounds and minimum size"""
        # Minimum size
        w = max(w, self.MIN_BOX_SIZE)
        h = max(h, self.MIN_BOX_SIZE)
        
        # Constrain to image bounds
        x = max(0, x)
        y = max(0, y)
        
        if x + w > self.image_width:
            if w <= self.image_width:
                x = self.image_width - w
            else:
                w = self.image_width
                x = 0
        
        if y + h > self.image_height:
            if h <= self.image_height:
                y = self.image_height - h
            else:
                h = self.image_height
                y = 0
        
        return x, y, w, h
    
    def _get_click_target(self, x: float, y: float) -> Tuple[str, Any]:
        """Determine what was clicked: ('handle', name), ('box', BBox), or ('empty', None)
        
        Only returns editable boxes for selection.
        When multiple boxes contain the click point, selects the smallest box (most precise).
        """
        # Check handles first (if a box is selected)
        selected_box = self.get_selected_box()
        if selected_box:
            handle = self._get_handle_at(x, y, selected_box)
            if handle:
                return ('handle', handle)
        
        # Collect all boxes that contain the click point
        # Check GT boxes first, then Pred boxes
        candidate_boxes = []
        
        for box in self.gt_boxes:
            if getattr(box, 'visible', True) and getattr(box, 'editable', True):
                if self._point_in_box(x, y, box):
                    candidate_boxes.append(box)
        
        for box in self.pred_boxes:
            if getattr(box, 'visible', True) and getattr(box, 'editable', True):
                if self._point_in_box(x, y, box):
                    candidate_boxes.append(box)
        
        # If no boxes contain the click point, return empty
        if not candidate_boxes:
            return ('empty', None)
        
        # Select the smallest box (by area) - most precise annotation
        # This ensures nested boxes are handled correctly
        smallest_box = min(candidate_boxes, key=lambda b: b.w * b.h)
        
        return ('box', smallest_box)
    
    def _get_handle_at(self, x: float, y: float, box: BBox) -> Optional[str]:
        """Check if point is on a handle, return handle name or None"""
        bx, by, bw, bh = box.x, box.y, box.w, box.h
        hs = self.HANDLE_SIZE
        
        positions = {
            'nw': (bx, by),
            'n':  (bx + bw/2, by),
            'ne': (bx + bw, by),
            'e':  (bx + bw, by + bh/2),
            'se': (bx + bw, by + bh),
            's':  (bx + bw/2, by + bh),
            'sw': (bx, by + bh),
            'w':  (bx, by + bh/2),
        }
        
        for name, (cx, cy) in positions.items():
            if abs(x - cx) <= hs and abs(y - cy) <= hs:
                return name
        
        return None
    
    def _point_in_box(self, x: float, y: float, box: BBox) -> bool:
        """Check if point is inside box"""
        return (box.x <= x <= box.x + box.w and 
                box.y <= y <= box.y + box.h)
    
    def _handle_key(self, e) -> None:
        """Handle keyboard events"""
        if not e.action.keydown:
            return
        
        # Get key name
        key_name = e.key.name if hasattr(e.key, 'name') else str(e.key)
        
        # Undo/Redo
        if key_name.lower() == 'z' and e.modifiers.ctrl and not e.modifiers.shift:
            self.undo()
            return
        if (key_name.lower() == 'y' and e.modifiers.ctrl) or (key_name.lower() == 'z' and e.modifiers.ctrl and e.modifiers.shift):
            self.redo()
            return
        
        # Zoom shortcuts (global, work regardless of selection)
        if not e.modifiers.ctrl and not e.modifiers.alt:
            # = or + for zoom in
            if key_name == 'Equal' or key_name == '=' or key_name == '+':
                self.zoom_in()
                return
            # - for zoom out
            if key_name == 'Minus' or key_name == '-':
                self.zoom_out()
                return
            # 0 for reset zoom
            if key_name == '0' or key_name == 'Digit0':
                self.reset_zoom()
                return
        
        # Cycle selection (prefer middle dot / backquote key)
        if not e.modifiers.ctrl and not e.modifiers.alt and not e.modifiers.shift:
            if key_name in ('·', '`', 'Backquote', 'Dead'):
                self._cycle_selection()
                return
        
        # Delete selected box
        if e.key.delete if hasattr(e.key, 'delete') else key_name == 'Delete':
            self._delete_selected()
            return
        if e.key.backspace if hasattr(e.key, 'backspace') else key_name == 'Backspace':
            self._delete_selected()
            return
        
        # Display toggle shortcuts (without modifiers)
        if not e.modifiers.ctrl and not e.modifiers.alt and not e.modifiers.shift:
            if key_name.lower() == 'q':
                # Toggle Show GT
                self.show_gt = not self.show_gt
                self._update_display()
                if self.on_display_change:
                    self.on_display_change(self.show_gt, self.show_pred)
                return
            if key_name.lower() == 'w':
                # Toggle Show Pred
                self.show_pred = not self.show_pred
                self._update_display()
                if self.on_display_change:
                    self.on_display_change(self.show_gt, self.show_pred)
                return
            if key_name.lower() == 'e':
                # Swap Editable
                self.swap_editable()
                return
            if key_name.lower() == 'r':
                # Clear Editable
                self.clear_editable()
                return
            if key_name.lower() == 't':
                # Activate Reference
                self.activate_reference()
                return
        
        # Class change shortcuts (without modifiers except shift)
        if not e.modifiers.ctrl and not e.modifiers.alt:
            class_keys = {'1': 0, '2': 1, '3': 2, '4': 3, 'Digit1': 0, 'Digit2': 1, 'Digit3': 2, 'Digit4': 3}
            if key_name in class_keys:
                self._change_class(class_keys[key_name])
                return
        
        # Box adjustment shortcuts (only when box is selected)
        if self.selected_box_id and not e.modifiers.ctrl and not e.modifiers.alt:
            # Determine if shift is pressed for uppercase behavior
            is_shift = e.modifiers.shift
            
            # Arrow keys for moving selected box
            arrow_left = e.key.arrow_left if hasattr(e.key, 'arrow_left') else key_name == 'ArrowLeft'
            arrow_right = e.key.arrow_right if hasattr(e.key, 'arrow_right') else key_name == 'ArrowRight'
            arrow_up = e.key.arrow_up if hasattr(e.key, 'arrow_up') else key_name == 'ArrowUp'
            arrow_down = e.key.arrow_down if hasattr(e.key, 'arrow_down') else key_name == 'ArrowDown'
            
            move_delta = 10 if is_shift else 1  # Shift for faster movement
            
            if arrow_left:
                self._move_selected(-move_delta, 0)
                return
            if arrow_right:
                self._move_selected(move_delta, 0)
                return
            if arrow_up:
                self._move_selected(0, -move_delta)
                return
            if arrow_down:
                self._move_selected(0, move_delta)
                return
            
            # Edge adjustments
            # a/A: left edge, s/S: top edge, d/D: right edge, f/F: bottom edge
            edge_map = {
                'a': ('left', 1 if is_shift else -1),   # a: left goes left (-1), A: left goes right (+1)
                's': ('top', 1 if is_shift else -1),    # s: top goes up (-1), S: top goes down (+1)
                'd': ('right', -1 if is_shift else 1),  # d: right goes right (+1), D: right goes left (-1)
                'f': ('bottom', -1 if is_shift else 1), # f: bottom goes down (+1), F: bottom goes up (-1)
            }
            
            # Corner adjustments
            # z/Z: nw corner, x/X: ne corner, c/C: se corner, v/V: sw corner
            corner_map = {
                'z': ('nw', 1 if is_shift else -1),   # z: expand, Z: shrink
                'x': ('ne', 1 if is_shift else -1),
                'c': ('se', 1 if is_shift else -1),
                'v': ('sw', 1 if is_shift else -1),
            }
            
            key_lower = key_name.lower()
            
            if key_lower in edge_map:
                edge, delta = edge_map[key_lower]
                self._adjust_edge(edge, delta)
                return
            
            if key_lower in corner_map:
                corner, delta = corner_map[key_lower]
                self._adjust_corner(corner, delta)
                return
    
    def _cycle_selection(self) -> None:
        """Cycle through editable boxes selection (both GT and Pred)"""
        # Collect all editable boxes
        editable_boxes = []
        for box in self.gt_boxes:
            if getattr(box, 'visible', True) and getattr(box, 'editable', True):
                editable_boxes.append(box)
        for box in self.pred_boxes:
            if getattr(box, 'visible', True) and getattr(box, 'editable', True):
                editable_boxes.append(box)
        
        if not editable_boxes:
            self.selected_box_id = None
            self._update_display()
            self._notify_change()
            return
        
        if not self.selected_box_id:
            # Select first box
            self.selected_box_id = editable_boxes[0].id
        else:
            # Find current index and move to next
            current_idx = -1
            for i, box in enumerate(editable_boxes):
                if box.id == self.selected_box_id:
                    current_idx = i
                    break
            
            next_idx = (current_idx + 1) % len(editable_boxes)
            self.selected_box_id = editable_boxes[next_idx].id
        
        self._update_display()
        self._notify_change()
    
    def _delete_selected(self) -> None:
        """Delete the selected box (must be editable)"""
        if not self.selected_box_id:
            return
        
        # Check if selected box is editable before deleting
        selected_box = self.get_selected_box()
        if not selected_box or not getattr(selected_box, 'editable', True):
            return
        
        # Remove from GT boxes
        self.gt_boxes = [b for b in self.gt_boxes if b.id != self.selected_box_id]
        # Remove from Pred boxes (in case it was moved there)
        self.pred_boxes = [b for b in self.pred_boxes if b.id != self.selected_box_id]
        
        self.selected_box_id = None
        
        self._save_history()
        self._notify_change()
        self._update_display()
    
    def _change_class(self, class_id: int) -> None:
        """Change class of selected box"""
        selected_box = self.get_selected_box()
        if not selected_box:
            # Just update current class for new boxes
            self.current_class = class_id
            return
        
        selected_box.class_id = class_id
        self._save_history()
        self._notify_change()
        self._update_display()
    
    def _move_selected(self, dx: int, dy: int) -> None:
        """Move selected box by delta pixels"""
        selected_box = self.get_selected_box()
        if not selected_box:
            return
        
        new_x = selected_box.x + dx
        new_y = selected_box.y + dy
        
        # Constrain to image bounds
        new_x = max(0, min(new_x, self.image_width - selected_box.w))
        new_y = max(0, min(new_y, self.image_height - selected_box.h))
        
        selected_box.x = new_x
        selected_box.y = new_y
        
        self._save_history()
        self._notify_change()
        self._update_display()
    
    def _adjust_edge(self, edge: str, delta: int) -> None:
        """Adjust a single edge by delta pixels"""
        selected_box = self.get_selected_box()
        if not selected_box:
            return
        
        x, y, w, h = selected_box.x, selected_box.y, selected_box.w, selected_box.h
        
        if edge == 'left':
            x += delta
            w -= delta
        elif edge == 'right':
            w += delta
        elif edge == 'top':
            y += delta
            h -= delta
        elif edge == 'bottom':
            h += delta
        
        x, y, w, h = self._constrain_box(x, y, w, h)
        
        selected_box.x = x
        selected_box.y = y
        selected_box.w = w
        selected_box.h = h
        
        self._save_history()
        self._notify_change()
        self._update_display()
    
    def _adjust_corner(self, corner: str, delta: int) -> None:
        """
        Adjust a corner by delta pixels.
        delta < 0: expand outward (corner moves away from center)
        delta > 0: shrink inward (corner moves toward center)
        
        nw (z/Z): adjust left edge and top edge
        ne (x/X): adjust right edge and top edge  
        se (c/C): adjust right edge and bottom edge
        sw (v/V): adjust left edge and bottom edge
        """
        selected_box = self.get_selected_box()
        if not selected_box:
            return
        
        x, y, w, h = selected_box.x, selected_box.y, selected_box.w, selected_box.h
        
        if corner == 'nw':
            # z: left goes left, top goes up (expand) -> x-=1, y-=1, w+=1, h+=1
            # Z: left goes right, top goes down (shrink) -> x+=1, y+=1, w-=1, h-=1
            x += delta
            y += delta
            w -= delta
            h -= delta
        elif corner == 'ne':
            # x: right goes right, top goes up (expand) -> y-=1, w+=1, h+=1
            # X: right goes left, top goes down (shrink) -> y+=1, w-=1, h-=1
            y += delta
            w -= delta
            h -= delta
        elif corner == 'se':
            # c: right goes right, bottom goes down (expand) -> w+=1, h+=1
            # C: right goes left, bottom goes up (shrink) -> w-=1, h-=1
            w -= delta
            h -= delta
        elif corner == 'sw':
            # v: left goes left, bottom goes down (expand) -> x-=1, w+=1, h+=1
            # V: left goes right, bottom goes up (shrink) -> x+=1, w-=1, h-=1
            x += delta
            w -= delta
            h -= delta
        
        x, y, w, h = self._constrain_box(x, y, w, h)
        
        selected_box.x = x
        selected_box.y = y
        selected_box.w = w
        selected_box.h = h
        
        self._save_history()
        self._notify_change()
        self._update_display()




