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
    """Snapshot of GT boxes state for undo/redo"""
    gt_boxes: List[BBox]
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
    
    # Available zoom levels (as multipliers)
    ZOOM_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def __init__(self, on_change: Callable[[List[BBox]], None] = None,
                 on_zoom_change: Callable[[float], None] = None):
        """
        Args:
            on_change: Callback when GT boxes change
            on_zoom_change: Callback when zoom level changes
        """
        self.on_change = on_change
        self.on_zoom_change = on_zoom_change
        
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
        
    def create_ui(self, container, fixed_width: int = 800, fixed_height: int = 600) -> None:
        """Create UI components in the specified container
        
        Args:
            container: Parent container
            fixed_width: Fixed width for the image area in pixels
            fixed_height: Fixed height for the image area in pixels
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
            # Main layout: image area + minimap on right
            with ui.row().classes('gap-4 items-start'):
                # Left: Main image with scrollbars
                with ui.column().classes('gap-0'):
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
                
                # Right: Minimap navigator
                with ui.column().classes('gap-1'):
                    ui.label('Navigator').classes('text-xs text-gray-500')
                    self.minimap_component = ui.interactive_image(
                        source='',
                        on_mouse=self._handle_minimap_mouse,
                        events=['mousedown', 'mouseup', 'mousemove'],
                        cross=False,
                        sanitize=False,
                    ).style(f'width: {self.minimap_width}px; height: {self.minimap_height}px; object-fit: contain;') \
                     .classes('border border-gray-300 rounded cursor-pointer')
            
            # Set up keyboard listener
            self.keyboard = ui.keyboard(on_key=self._handle_key, ignore=['input', 'select', 'textarea'])
    
    def load_image(self, image_path: str) -> None:
        """Load image and get its dimensions"""
        from pathlib import Path
        self.image_path = image_path
        path = Path(image_path)
        
        if path.exists():
            self.image_width, self.image_height = get_image_size(path)
            if self.image_component:
                self.image_component.set_source(image_path)
            # Also load into minimap
            if hasattr(self, 'minimap_component') and self.minimap_component:
                self.minimap_component.set_source(image_path)
            # Reset zoom when loading new image
            self.reset_zoom()
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
            self.selected_box_id = state.selected_id
            self._update_display()
            self._notify_change()
            return True
        return False
    
    def get_selected_box(self) -> Optional[BBox]:
        """Get currently selected box"""
        if self.selected_box_id:
            for box in self.gt_boxes:
                if box.id == self.selected_box_id:
                    return box
        return None
    
    # ==================== Zoom Methods ====================
    
    def _get_max_pan(self) -> tuple:
        """Get maximum pan values based on current zoom
        
        Returns:
            (max_pan_x, max_pan_y) in image coordinates
        """
        if self.zoom <= 1.0:
            return (0, 0)
        visible_width = self.view_width / self.zoom
        visible_height = self.view_height / self.zoom
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
            self.h_scrollbar.set_visibility(True)
            self.v_scrollbar.set_visibility(True)
            
            max_pan_x, max_pan_y = self._get_max_pan()
            
            if max_pan_x > 0:
                h_value = (self.pan_x / max_pan_x) * 100
                self.h_scrollbar.set_value(h_value)
            
            if max_pan_y > 0:
                v_value = (self.pan_y / max_pan_y) * 100
                self.v_scrollbar.set_value(v_value)
    
    def set_zoom(self, zoom: float, focus_point: tuple = None) -> None:
        """Set zoom level (1.0 = 100%)
        
        Args:
            zoom: Target zoom level
            focus_point: (x, y) in image coordinates to keep centered. If None, auto-determine.
        """
        new_zoom = max(1.0, min(zoom, 10.0))
        
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
            # At old zoom: screen_center = (pan_x + view_width/(2*old_zoom), pan_y + view_height/(2*old_zoom))
            # We want focus_point to remain at screen center after zoom change
            # New pan: pan_x = focus_x - view_width/(2*new_zoom)
            self.pan_x = focus_x - self.view_width / (2 * self.zoom)
            self.pan_y = focus_y - self.view_height / (2 * self.zoom)
        
        self._constrain_pan()
        self._apply_transform()
        self._update_scrollbars()
        
        if self.on_zoom_change:
            self.on_zoom_change(self.zoom)
    
    def _get_zoom_focus_point(self) -> tuple:
        """Get the focus point for zooming (in image coordinates)
        
        Always focuses on the center of the current view.
        
        Returns:
            (x, y) - center of current view in image coordinates
        """
        # Always focus on current view center
        if self.zoom > 1.0:
            # View center in image coordinates
            cx = self.pan_x + self.view_width / (2 * self.zoom)
            cy = self.pan_y + self.view_height / (2 * self.zoom)
        else:
            # At 1x zoom, center of image (or view if image is smaller)
            cx = min(self.image_width, self.view_width) / 2
            cy = min(self.image_height, self.view_height) / 2
        
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
    
    def zoom_in(self) -> None:
        """Zoom in by 1 level"""
        self.set_zoom(self.zoom + 1)
    
    def zoom_out(self) -> None:
        """Zoom out by 1 level"""
        self.set_zoom(self.zoom - 1)
    
    def _apply_transform(self) -> None:
        """Apply CSS transform for zoom and pan"""
        if not hasattr(self, 'transform_container') or not self.transform_container:
            return
        
        # Use translate then scale (CSS applies right-to-left)
        # We want: first translate by -pan, then scale
        # In CSS: transform: scale(z) translate(-px, -py)
        # This means: scale first, then translate (in scaled space)
        # Actually we need: translate(-pan*zoom, -pan*zoom) scale(zoom)
        # Which gives us the effect of panning in image space then scaling
        
        # Simpler approach: translate in unscaled space, then scale
        translate_x = -self.pan_x * self.zoom
        translate_y = -self.pan_y * self.zoom
        
        transform = f'translate({translate_x}px, {translate_y}px) scale({self.zoom})'
        self.transform_container.style(f'transform: {transform}; transform-origin: 0 0;')
        
        # Update minimap viewport indicator
        self._update_minimap()
    
    def _update_minimap(self) -> None:
        """Update minimap with viewport indicator rectangle"""
        if not hasattr(self, 'minimap_component') or not self.minimap_component:
            return
        if self.image_width <= 0 or self.image_height <= 0:
            return
        
        # The minimap shows the FULL IMAGE, but scaled down to fit minimap_width x minimap_height
        # The main view shows only a portion of the image (view_width x view_height at zoom=1x)
        # 
        # We need to calculate the scale factor between image coords and minimap display coords
        # Minimap displays the full image, so:
        #   minimap_scale_x = minimap_width / image_width
        #   minimap_scale_y = minimap_height / image_height
        # But since we use object-fit: contain, the actual scale is the min of these
        
        # For simplicity, let's calculate based on the image aspect ratio
        img_aspect = self.image_width / self.image_height
        minimap_aspect = self.minimap_width / self.minimap_height
        
        if img_aspect > minimap_aspect:
            # Image is wider - width is the constraint
            minimap_img_scale = self.minimap_width / self.image_width
        else:
            # Image is taller - height is the constraint
            minimap_img_scale = self.minimap_height / self.image_height
        
        # The viewport in the main view (in image coordinates):
        # Position: (pan_x, pan_y), Size: (view_width/zoom, view_height/zoom)
        viewport_x = self.pan_x
        viewport_y = self.pan_y
        viewport_w = self.view_width / self.zoom
        viewport_h = self.view_height / self.zoom
        
        # Convert to minimap coordinates
        rect_x = viewport_x * minimap_img_scale
        rect_y = viewport_y * minimap_img_scale
        rect_w = viewport_w * minimap_img_scale
        rect_h = viewport_h * minimap_img_scale
        
        # #region agent log
        import json; open('/home/yangxinyu/Test/Projects/refiner/.cursor/debug.log', 'a').write(json.dumps({"location": "components.py:_update_minimap", "message": "minimap_update", "data": {"image_size": [self.image_width, self.image_height], "minimap_size": [self.minimap_width, self.minimap_height], "minimap_img_scale": minimap_img_scale, "viewport_img": [viewport_x, viewport_y, viewport_w, viewport_h], "rect_minimap": [rect_x, rect_y, rect_w, rect_h], "zoom": self.zoom}, "timestamp": __import__('time').time()*1000, "sessionId": "debug-session", "hypothesisId": "G"}) + '\n')
        # #endregion
        
        # Only show viewport rect when zoomed in
        if self.zoom > 1.0:
            svg_content = f'''
                <rect x="{rect_x}" y="{rect_y}" width="{rect_w}" height="{rect_h}"
                      fill="rgba(59, 130, 246, 0.2)" stroke="#3b82f6" stroke-width="2"
                      pointer-events="all" cursor="move" />
            '''
        else:
            svg_content = ''
        
        self.minimap_component.set_content(svg_content)
    
    def _handle_minimap_mouse(self, e) -> None:
        """Handle mouse events on minimap"""
        event_type = e.type
        
        # Get click position in minimap coordinates (these are image coords scaled down)
        x = e.image_x if hasattr(e, 'image_x') else 0
        y = e.image_y if hasattr(e, 'image_y') else 0
        
        if event_type == 'mousedown':
            self.minimap_dragging = True
            # Center the viewport on click position
            self._minimap_set_center(x, y)
        elif event_type == 'mousemove' and self.minimap_dragging:
            self._minimap_set_center(x, y)
        elif event_type == 'mouseup':
            self.minimap_dragging = False
    
    def _minimap_set_center(self, minimap_x: float, minimap_y: float) -> None:
        """Set the main view pan so that the viewport is centered at the given minimap position
        
        Args:
            minimap_x, minimap_y: Coordinates from interactive_image mouse event (in original image coords)
        """
        if self.zoom <= 1.0:
            return
        
        # The minimap's interactive_image returns coordinates in the ORIGINAL IMAGE coordinate system
        # (this is how ui.interactive_image works - it returns image coords regardless of display size)
        # So minimap_x, minimap_y are already in image coordinates!
        
        # Set pan so that (minimap_x, minimap_y) is at the center of the viewport
        visible_w = self.view_width / self.zoom
        visible_h = self.view_height / self.zoom
        
        self.pan_x = minimap_x - visible_w / 2
        self.pan_y = minimap_y - visible_h / 2
        
        # #region agent log
        import json; open('/home/yangxinyu/Test/Projects/refiner/.cursor/debug.log', 'a').write(json.dumps({"location": "components.py:_minimap_set_center", "message": "minimap_click", "data": {"click_img_coords": [minimap_x, minimap_y], "visible_size": [visible_w, visible_h], "new_pan": [self.pan_x, self.pan_y]}, "timestamp": __import__('time').time()*1000, "sessionId": "debug-session", "hypothesisId": "G"}) + '\n')
        # #endregion
        
        self._constrain_pan()
        self._apply_transform()
        self._update_scrollbars()
    
    def _constrain_pan(self) -> None:
        """Constrain pan to keep image visible in viewport"""
        if self.zoom <= 1.0:
            self.pan_x = 0
            self.pan_y = 0
            return
        
        # Pan is in image coordinates
        # At zoom Z, visible area in image coords is (view_width/Z, view_height/Z)
        # Maximum pan is image_size - visible_area
        visible_width = self.view_width / self.zoom
        visible_height = self.view_height / self.zoom
        
        max_pan_x = max(0, self.image_width - visible_width)
        max_pan_y = max(0, self.image_height - visible_height)
        
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))
    
    # ==================== Private Methods ====================
    
    def _save_history(self) -> None:
        """Save current state to history"""
        # Remove any redo states
        self.history = self.history[:self.history_index + 1]
        
        # Add new state
        state = HistoryState(
            gt_boxes=deepcopy(self.gt_boxes),
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
        """Update SVG overlay content"""
        if not self.image_component:
            return
        
        svg_parts = []
        
        # Render GT boxes
        if self.show_gt:
            for box in self.gt_boxes:
                svg_parts.append(self._render_box(box))
        
        # Render Pred boxes
        if self.show_pred:
            for box in self.pred_boxes:
                svg_parts.append(self._render_box(box))
        
        # Render handles for selected box
        selected_box = self.get_selected_box()
        if selected_box and self.show_gt:
            svg_parts.append(self._render_handles(selected_box))
        
        self.image_component.set_content(''.join(svg_parts))
    
    def _render_box(self, box: BBox) -> str:
        """Render a single bounding box as SVG"""
        is_selected = box.id == self.selected_box_id
        source = box.source.value if isinstance(box.source, BoxSource) else box.source
        
        color = self.COLORS[source]['selected' if is_selected else 'normal']
        stroke_width = 3 if is_selected else 2
        dash_array = '' if source == 'gt' else '5,5'
        
        # Box rectangle
        rect_attrs = f'x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}"'
        rect_style = f'fill="none" stroke="{color}" stroke-width="{stroke_width}"'
        if dash_array:
            rect_style += f' stroke-dasharray="{dash_array}"'
        
        svg = f'<rect {rect_attrs} {rect_style} data-box-id="{box.id}"/>'
        
        # Label text
        if source == 'gt':
            label = f"{box.class_id}"
        else:
            # For pred boxes, show class and confidence if available
            label = f"{box.class_id}"
        
        # Background for label
        label_x = box.x
        label_y = box.y - 4
        if label_y < 15:
            label_y = box.y + 15
        
        svg += f'''<text x="{label_x + 2}" y="{label_y}" 
            font-size="12" font-family="Arial" font-weight="bold"
            fill="{color}" stroke="white" stroke-width="0.5">{label}</text>'''
        
        return svg
    
    def _render_handles(self, box: BBox) -> str:
        """Render resize handles for selected box"""
        x, y, w, h = box.x, box.y, box.w, box.h
        hs = self.HANDLE_SIZE
        
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
                f'<rect x="{cx - hs/2}" y="{cy - hs/2}" '
                f'width="{hs}" height="{hs}" '
                f'fill="white" stroke="#ef4444" stroke-width="1" '
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
            elif self.zoom <= 1.0:
                # Start creating new box (only in 1x zoom)
                # In zoomed mode, use scrollbars to pan instead of drag (drag has coordinate issues)
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
        """Determine what was clicked: ('handle', name), ('box', BBox), or ('empty', None)"""
        # Check handles first (if a box is selected)
        selected_box = self.get_selected_box()
        if selected_box:
            handle = self._get_handle_at(x, y, selected_box)
            if handle:
                return ('handle', handle)
        
        # Check GT boxes (reverse order for top-most first)
        for box in reversed(self.gt_boxes):
            if self._point_in_box(x, y, box):
                return ('box', box)
        
        return ('empty', None)
    
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
        
        # Tab - cycle selection
        if e.key.tab if hasattr(e.key, 'tab') else key_name == 'Tab':
            self._cycle_selection()
            return
        
        # Delete selected box
        if e.key.delete if hasattr(e.key, 'delete') else key_name == 'Delete':
            self._delete_selected()
            return
        if e.key.backspace if hasattr(e.key, 'backspace') else key_name == 'Backspace':
            self._delete_selected()
            return
        
        # Class change shortcuts (without modifiers except shift)
        if not e.modifiers.ctrl and not e.modifiers.alt:
            class_keys = {'q': 0, 'w': 1, 'e': 2, 'r': 3}
            if key_name.lower() in class_keys:
                self._change_class(class_keys[key_name.lower()])
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
        """Cycle through GT boxes selection"""
        if not self.gt_boxes:
            return
        
        if not self.selected_box_id:
            # Select first box
            self.selected_box_id = self.gt_boxes[0].id
        else:
            # Find current index and move to next
            current_idx = -1
            for i, box in enumerate(self.gt_boxes):
                if box.id == self.selected_box_id:
                    current_idx = i
                    break
            
            next_idx = (current_idx + 1) % len(self.gt_boxes)
            self.selected_box_id = self.gt_boxes[next_idx].id
        
        self._update_display()
    
    def _delete_selected(self) -> None:
        """Delete the selected box"""
        if not self.selected_box_id:
            return
        
        self.gt_boxes = [b for b in self.gt_boxes if b.id != self.selected_box_id]
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
