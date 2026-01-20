"""
Dashboard page - Analysis control panel with visualization
"""
import asyncio
import tempfile
import os
import time
from pathlib import Path
from nicegui import ui, app
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from ..state import app_state
from ..models import IssueType, IssueItem, ClassMapping
from ..core.file_manager import validate_paths, backup_folder, validate_output_path, ensure_output_structure
from ..core.analyzer import CleanlabAnalyzer
from ..core.yolo_utils import read_yolo_label, get_image_size


class DashboardPage:
    """Dashboard page for configuring and running Cleanlab analysis"""
    
    def __init__(self):
        self.progress_bar = None
        self.progress_label = None
        self.run_button = None
        self.goto_button = None
        
        # Path input fields
        self.images_input = None
        self.gt_input = None
        self.pred_input = None
        self.output_input = None
        self.human_verified_input = None
        self.classes_input = None
        
        # Validation labels
        self.images_status = None
        self.gt_status = None
        self.pred_status = None
        self.output_status = None
        
        # TopK input
        self.topk_input = None
        self.refresh_button = None
        
        # Checkboxes
        self.backup_checkbox = None
        self.overlooked_checkbox = None
        self.swapped_checkbox = None
        self.badloc_checkbox = None
        
        # Result containers and count labels
        self.overlooked_container = None
        self.swapped_container = None
        self.badloc_container = None
        self.overlooked_count = None
        self.swapped_count = None
        self.badloc_count = None
        
        # Visualization
        self.viz_container = None
        self.viz_image = None
        self.viz_info = None
        self.current_viz_file = None  # Track temp file for cleanup
        
        # Currently selected item
        self.selected_item = None
    
    def create(self):
        """Create the dashboard page"""
        ui.add_head_html('''
        <style>
            .issue-item { transition: background-color 0.15s; }
            .issue-item:hover { background-color: #e5e7eb; }
            .issue-item-selected { background-color: #dbeafe !important; }
            .issue-list { scrollbar-width: thin; }
            .viz-placeholder { 
                display: flex; 
                align-items: center; 
                justify-content: center;
                background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
                color: #627d98;
            }
        </style>
        ''')
        
        with ui.column().classes('w-full min-h-screen bg-gray-100'):
            # Header
            with ui.row().classes('w-full bg-white shadow px-6 py-3 items-center gap-3'):
                ui.icon('auto_fix_high', size='28px').classes('text-indigo-600')
                ui.label('Annotation Refiner').classes('text-xl font-bold text-gray-800')
            
            # Main content - horizontal layout
            with ui.row().classes('w-full p-4 gap-4 flex-nowrap'):
                # Left: Configuration panel (narrow)
                self._create_config_panel()
                
                # Middle: Issue lists (3 columns)
                self._create_issue_lists()
                
                # Right: Visualization panel
                self._create_viz_panel()
    
    def _create_config_panel(self):
        """Create configuration panel"""
        with ui.card().classes('w-72 flex-shrink-0 shadow'):
            with ui.column().classes('w-full p-4 gap-3'):
                ui.label('Configuration').classes('text-base font-bold text-gray-700')
                
                # Images path
                ui.label('Images Path').classes('text-xs font-medium text-gray-500 mt-1')
                with ui.row().classes('w-full items-center gap-1'):
                    self.images_input = ui.input().classes('flex-grow').props('dense outlined size=small')
                    self.images_status = ui.label('').classes('text-xs whitespace-nowrap')
                
                # GT Labels path
                ui.label('GT Labels Path').classes('text-xs font-medium text-gray-500')
                with ui.row().classes('w-full items-center gap-1'):
                    self.gt_input = ui.input().classes('flex-grow').props('dense outlined size=small')
                    self.gt_status = ui.label('').classes('text-xs whitespace-nowrap')
                
                # Pred Labels path
                ui.label('Pred Labels Path').classes('text-xs font-medium text-gray-500')
                with ui.row().classes('w-full items-center gap-1'):
                    self.pred_input = ui.input().classes('flex-grow').props('dense outlined size=small')
                    self.pred_status = ui.label('').classes('text-xs whitespace-nowrap')
                
                # Output Path
                ui.label('Output Path').classes('text-xs font-medium text-gray-500')
                with ui.row().classes('w-full items-center gap-1'):
                    self.output_input = ui.input(
                        value='/home/yangxinyu/Test/Data/internalVideos_fireRelated_keyFrameAnnotations_verifying'
                    ).classes('flex-grow').props('dense outlined size=small')
                    self.output_status = ui.label('').classes('text-xs whitespace-nowrap')
                
                # Human Verified Annotation Path
                ui.label('Human Verified Annotation Path (opt)').classes('text-xs font-medium text-gray-500')
                self.human_verified_input = ui.input().classes('w-full').props('dense outlined size=small')
                
                # Classes file
                ui.label('Classes File (opt)').classes('text-xs font-medium text-gray-500')
                self.classes_input = ui.input().classes('w-full').props('dense outlined size=small')
                
                ui.separator().classes('my-2')
                
                # Run button
                self.run_button = ui.button(
                    'RUN ANALYSIS',
                    on_click=self._run_analysis,
                    icon='play_arrow'
                ).classes('w-full').props('color=primary')
                
                # Progress
                self.progress_label = ui.label('').classes('text-xs text-gray-500')
                self.progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                self.progress_bar.visible = False
                
                ui.separator().classes('my-2')
                
                # Backup option - default to False
                self.backup_checkbox = ui.checkbox(
                    'Backup GT folder before analysis',
                    value=False
                ).classes('text-xs')
        
        # Bind events
        self.images_input.on('change', self._on_path_change)
        self.gt_input.on('change', self._on_path_change)
        self.pred_input.on('change', self._on_path_change)
        self.output_input.on('change', self._on_output_path_change)
        self.human_verified_input.on('change', self._on_output_path_change)
        self.classes_input.on('change', self._on_classes_change)
        
        # Initialize output path in config
        if self.output_input.value:
            app_state.config.output_path = self.output_input.value
        if self.human_verified_input.value:
            app_state.config.human_verified_path = self.human_verified_input.value
    
    def _create_issue_lists(self):
        """Create the three issue list columns"""
        with ui.column().classes('flex-shrink-0 gap-2'):
            # Three issue columns (header row aligned with config panel)
            with ui.row().classes('gap-2'):
                # Overlooked
                with ui.card().classes('w-56 shadow').style('border-left: 4px solid #f59e0b'):
                    with ui.column().classes('w-full p-2 gap-1'):
                        with ui.row().classes('items-center justify-between'):
                            self.overlooked_checkbox = ui.checkbox('Overlooked', value=True).classes('text-sm')
                            self.overlooked_count = ui.badge('0').props('color=orange')
                        self.overlooked_container = ui.scroll_area().classes('w-full h-80 issue-list')
                
                # Swapped
                with ui.card().classes('w-56 shadow').style('border-left: 4px solid #ef4444'):
                    with ui.column().classes('w-full p-2 gap-1'):
                        with ui.row().classes('items-center justify-between'):
                            self.swapped_checkbox = ui.checkbox('Swapped', value=True).classes('text-sm')
                            self.swapped_count = ui.badge('0').props('color=red')
                        self.swapped_container = ui.scroll_area().classes('w-full h-80 issue-list')
                
                # Bad Located
                with ui.card().classes('w-56 shadow').style('border-left: 4px solid #8b5cf6'):
                    with ui.column().classes('w-full p-2 gap-1'):
                        with ui.row().classes('items-center justify-between'):
                            self.badloc_checkbox = ui.checkbox('Bad Located', value=True).classes('text-sm')
                            self.badloc_count = ui.badge('0').props('color=purple')
                        self.badloc_container = ui.scroll_area().classes('w-full h-80 issue-list')
            
            # TopK control row (below issue lists)
            with ui.row().classes('items-center gap-2 mt-2'):
                ui.label('TopK:').classes('text-xs text-gray-500')
                self.topk_input = ui.number(value=10, min=1, max=1000, step=1).classes('w-16').props('dense outlined size=small')
                self.refresh_button = ui.button(icon='refresh', on_click=self._on_refresh_topk).props('flat dense size=sm')
                self.refresh_button.disable()
            
            # Go to annotation button
            with ui.row().classes('justify-end mt-2'):
                self.goto_button = ui.button(
                    'Go to Annotation Tool',
                    on_click=self._goto_annotation,
                    icon='edit'
                ).props('color=positive')
                self.goto_button.disable()
        
        # Bind checkbox events
        self.overlooked_checkbox.on('change', lambda e: setattr(app_state, 'selected_overlooked', e.value))
        self.swapped_checkbox.on('change', lambda e: setattr(app_state, 'selected_swapped', e.value))
        self.badloc_checkbox.on('change', lambda e: setattr(app_state, 'selected_bad_located', e.value))
    
    def _create_viz_panel(self):
        """Create visualization panel"""
        with ui.card().classes('flex-grow shadow'):
            with ui.column().classes('w-full h-full p-3 gap-2'):
                ui.label('Visualization').classes('text-base font-bold text-gray-700')
                self.viz_info = ui.label('Click an issue to visualize').classes('text-xs text-gray-500')
                
                # Fixed size container for visualization
                self.viz_container = ui.column().classes('w-full flex-grow items-center justify-center')
                with self.viz_container:
                    # Placeholder
                    with ui.element('div').classes('w-full h-80 viz-placeholder rounded'):
                        ui.label('Select an issue from the list').classes('text-lg')
    
    def _on_path_change(self, e=None):
        """Handle path input change"""
        app_state.config.images_path = self.images_input.value or ''
        app_state.config.gt_labels_path = self.gt_input.value or ''
        app_state.config.pred_labels_path = self.pred_input.value or ''
        self._validate_paths()
    
    def _on_output_path_change(self, e=None):
        """Handle output path input change"""
        app_state.config.output_path = self.output_input.value or ''
        app_state.config.human_verified_path = self.human_verified_input.value or ''
        self._validate_output_path()
    
    def _validate_paths(self):
        """Validate all paths and update status labels"""
        images_path = self.images_input.value or ''
        gt_path = self.gt_input.value or ''
        pred_path = self.pred_input.value or ''
        
        self.images_status.text = ''
        self.gt_status.text = ''
        self.pred_status.text = ''
        
        if not images_path or not gt_path or not pred_path:
            return
        
        result = validate_paths(images_path, gt_path, pred_path)
        app_state.path_validation = result
        
        if result['valid']:
            self.images_status.text = f'{result["images_count"]} imgs'
            self.images_status.classes(remove='text-red-500', add='text-green-600')
            self.gt_status.text = f'{result["gt_count"]} files'
            self.gt_status.classes(remove='text-red-500', add='text-green-600')
            self.pred_status.text = f'{result["pred_count"]} files'
            self.pred_status.classes(remove='text-red-500', add='text-green-600')
        else:
            for error in result['errors']:
                if 'Images' in error:
                    self.images_status.text = 'Not found'
                    self.images_status.classes(remove='text-green-600', add='text-red-500')
                elif 'GT' in error:
                    self.gt_status.text = 'Not found'
                    self.gt_status.classes(remove='text-green-600', add='text-red-500')
                elif 'Pred' in error:
                    self.pred_status.text = 'Not found'
                    self.pred_status.classes(remove='text-green-600', add='text-red-500')
    
    def _validate_output_path(self):
        """Validate output path and check for conflicts"""
        output_path = self.output_input.value or ''
        gt_path = self.gt_input.value or ''
        pred_path = self.pred_input.value or ''
        
        self.output_status.text = ''
        
        if not output_path:
            return
        
        status, message = validate_output_path(output_path, gt_path, pred_path)
        
        if status == "error":
            self.output_status.text = 'Required'
            self.output_status.classes(remove='text-green-600 text-yellow-600', add='text-red-500')
        elif status == "warning":
            self.output_status.text = 'Warning'
            self.output_status.classes(remove='text-green-600 text-red-500', add='text-yellow-600')
            if message:
                ui.notify(message, type='warning', timeout=3000)
        else:
            self.output_status.text = 'OK'
            self.output_status.classes(remove='text-red-500 text-yellow-600', add='text-green-600')
    
    def _on_classes_change(self, e=None):
        """Handle classes file change"""
        app_state.config.classes_file = self.classes_input.value or ''
        if self.classes_input.value and Path(self.classes_input.value).exists():
            try:
                app_state.class_mapping = ClassMapping.from_file(self.classes_input.value)
            except Exception as ex:
                ui.notify(f'Failed to load classes file: {ex}', type='warning')
    
    def _on_refresh_topk(self):
        """Handle TopK refresh"""
        app_state.config.top_k = int(self.topk_input.value)
        if app_state.analysis_complete:
            self._update_results_display()
            ui.notify(f'Updated to Top {app_state.config.top_k}', type='info')
    
    async def _run_analysis(self):
        """Run Cleanlab analysis"""
        # Validate Output Path is required
        if not app_state.config.output_path or not app_state.config.output_path.strip():
            ui.notify('Output Path is required. Please set Output Path before running analysis.', 
                     type='negative', timeout=5000)
            return
        
        self._validate_paths()
        if not app_state.path_validation.get('valid', False):
            ui.notify('Please fix path errors before running analysis', type='negative')
            return
        
        # Record start time
        start_time = time.time()
        
        self.run_button.disable()
        self.progress_bar.visible = True
        app_state.is_analyzing = True
        app_state.reset_analysis()
        
        try:
            # Ensure output directory structure exists
            from ..core.yolo_utils import collect_image_paths
            
            self._update_progress('Preparing output directory...', 0.01)
            all_image_rel_paths = collect_image_paths(Path(app_state.config.images_path))
            ensure_output_structure(app_state.config.output_path, all_image_rel_paths)
            
            # Backup only if checkbox is checked
            if self.backup_checkbox.value:
                self._update_progress('Backing up GT folder...', 0.02)
                try:
                    backup_path = backup_folder(app_state.config.gt_labels_path)
                    ui.notify(f'Backup created: {Path(backup_path).name}', type='positive')
                except Exception as ex:
                    ui.notify(f'Backup failed: {ex}', type='warning')
            
            analyzer = CleanlabAnalyzer(
                images_path=app_state.config.images_path,
                pred_labels_path=app_state.config.pred_labels_path,
                gt_labels_path=app_state.config.gt_labels_path,
                output_path=app_state.config.output_path,
                human_verified_path=app_state.config.human_verified_path,
                progress_callback=self._update_progress
            )
            
            await asyncio.get_event_loop().run_in_executor(None, analyzer.prepare_data)
            
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: analyzer.analyze(top_k=1000)
            )
            
            app_state.results.overlooked = results[IssueType.OVERLOOKED]
            app_state.results.swapped = results[IssueType.SWAPPED]
            app_state.results.bad_located = results[IssueType.BAD_LOCATED]
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            
            if minutes > 0:
                time_str = f'{minutes}m {seconds}s'
            else:
                time_str = f'{seconds:.3f}s'
            
            app_state.analysis_complete = True
            self._update_progress('Analysis complete!', 1.0)
            self._update_results_display()
            
            self.goto_button.enable()
            self.refresh_button.enable()
            
            # Show counts for each category (these are stored with top_k=1000 from cleanlab)
            n_overlooked = len(app_state.results.overlooked)
            n_swapped = len(app_state.results.swapped)
            n_badloc = len(app_state.results.bad_located)
            ui.notify(
                f'Analysis complete: {n_overlooked} overlooked, {n_swapped} swapped, {n_badloc} bad located',
                type='positive'
            )
            
            # Log total elapsed time
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f'Total analysis time: {elapsed_time:.3f} seconds ({time_str})')
            
        except Exception as ex:
            # Calculate elapsed time even on error
            elapsed_time = time.time() - start_time
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f'Analysis failed after {elapsed_time:.3f} seconds: {ex}')
            
            ui.notify(f'Analysis failed: {ex}', type='negative')
            import traceback
            traceback.print_exc()
        finally:
            app_state.is_analyzing = False
            self.run_button.enable()
    
    def _update_progress(self, message: str, percentage: float):
        """Update progress display"""
        if self.progress_label:
            self.progress_label.text = message
        if self.progress_bar:
            self.progress_bar.value = percentage
    
    def _update_results_display(self):
        """Update results tables"""
        top_k = int(self.topk_input.value) if self.topk_input.value else 10
        
        overlooked_items = app_state.results.overlooked[:top_k]
        swapped_items = app_state.results.swapped[:top_k]
        badloc_items = app_state.results.bad_located[:top_k]
        
        self.overlooked_count.text = str(len(overlooked_items))
        self.swapped_count.text = str(len(swapped_items))
        self.badloc_count.text = str(len(badloc_items))
        
        self._rebuild_list(self.overlooked_container, overlooked_items, 'orange', IssueType.OVERLOOKED)
        self._rebuild_list(self.swapped_container, swapped_items, 'red', IssueType.SWAPPED)
        self._rebuild_list(self.badloc_container, badloc_items, 'purple', IssueType.BAD_LOCATED)
    
    def _rebuild_list(self, container, items, color, issue_type):
        """Rebuild a results list"""
        container.clear()
        
        with container:
            if not items:
                ui.label('No issues found').classes('text-gray-400 text-xs p-2 italic')
            else:
                for i, item in enumerate(items):
                    path_parts = Path(item.image_path).parts
                    short_path = '/'.join(path_parts[-2:]) if len(path_parts) > 2 else item.image_path
                    
                    with ui.row().classes(f'w-full items-center py-1 px-2 issue-item rounded cursor-pointer').on(
                        'click', lambda e, it=item, idx=i: self._on_item_click(it, idx)
                    ):
                        ui.label(f'{i+1}.').classes('w-5 text-gray-400 text-xs')
                        ui.label(short_path).classes('flex-grow text-xs truncate text-gray-700')
                        ui.label(f'{item.score:.3f}').classes(f'text-xs text-{color}-600 font-mono')
    
    def _on_item_click(self, item: IssueItem, index: int):
        """Handle item click - generate and show visualization"""
        self.selected_item = item
        self.viz_info.text = f'{item.issue_type.value}: {item.image_path}'
        
        # Clean up previous temp file
        if self.current_viz_file and os.path.exists(self.current_viz_file):
            try:
                os.remove(self.current_viz_file)
            except:
                pass
        
        # Generate visualization
        try:
            viz_path = self._generate_visualization(item)
            self.current_viz_file = viz_path
            
            # Update viz container
            self.viz_container.clear()
            with self.viz_container:
                ui.image(viz_path).classes('max-w-full max-h-96 object-contain')
        except Exception as ex:
            self.viz_container.clear()
            with self.viz_container:
                ui.label(f'Failed to visualize: {ex}').classes('text-red-500')
    
    def _generate_visualization(self, item: IssueItem) -> str:
        """Generate visualization image and return temp file path"""
        images_dir = Path(app_state.config.images_path)
        gt_dir = Path(app_state.config.gt_labels_path)
        pred_dir = Path(app_state.config.pred_labels_path)
        
        rel_path = Path(item.image_path)
        img_path = images_dir / rel_path
        gt_label_path = gt_dir / rel_path.with_suffix('.txt')
        pred_label_path = pred_dir / rel_path.with_suffix('.txt')
        
        # Load image
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # Load boxes
        gt_boxes = read_yolo_label(gt_label_path, img_w, img_h, has_confidence=False)
        pred_boxes = read_yolo_label(pred_label_path, img_w, img_h, has_confidence=True)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.imshow(img)
        
        # Draw GT boxes
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box['bbox']
            w, h = x2 - x1, y2 - y1
            is_issue = (i == item.box_index and item.issue_type in [IssueType.SWAPPED, IssueType.BAD_LOCATED])
            color = 'red' if is_issue else 'lime'
            lw = 3 if is_issue else 2
            rect = patches.Rectangle((x1, y1), w, h, linewidth=lw, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            label = f"GT:{box['class_id']}" + (" [ISSUE]" if is_issue else "")
            ax.text(x1, y1 - 5, label, color=color, fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Draw pred boxes
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = box['bbox']
            w, h = x2 - x1, y2 - y1
            conf = box.get('confidence', 0)
            is_issue = (i == item.box_index and item.issue_type == IssueType.OVERLOOKED)
            color = 'orange' if is_issue else 'deepskyblue'
            lw = 3 if is_issue else 1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=lw, edgecolor=color, facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x2, y2 + 12, f"P:{box['class_id']}({conf:.2f})", color=color, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.set_title(f"{item.issue_type.value.upper()} | Score: {item.score:.4f}", fontsize=11)
        ax.axis('off')
        plt.tight_layout()
        
        # Save to temp file
        fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='refiner_viz_')
        os.close(fd)
        plt.savefig(temp_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return temp_path
    
    def _goto_annotation(self):
        """Navigate to annotation page"""
        # Ensure checkbox states are synced to app_state (fix selection logic)
        app_state.selected_overlooked = self.overlooked_checkbox.value
        app_state.selected_swapped = self.swapped_checkbox.value
        app_state.selected_bad_located = self.badloc_checkbox.value
        
        # Get topK value and apply it when building annotation queue
        top_k = int(self.topk_input.value) if self.topk_input.value else 10
        
        # Build annotation queue based on selected types
        app_state.annotation_queue = app_state.get_selected_issues(top_k=top_k)
        app_state.current_annotation_index = 0
        
        if not app_state.annotation_queue:
            ui.notify('No issues selected. Please select at least one issue type.', type='warning')
            return
        
        # Show summary of selected types
        selected_types = []
        if app_state.selected_overlooked:
            selected_types.append('Overlooked')
        if app_state.selected_swapped:
            selected_types.append('Swapped')
        if app_state.selected_bad_located:
            selected_types.append('Bad Located')
        
        type_str = ', '.join(selected_types)
        ui.notify(f'Starting with {len(app_state.annotation_queue)} samples from {type_str} (TopK={top_k})', type='info')
        ui.navigate.to('/annotator')


def create_dashboard():
    """Create dashboard page"""
    page = DashboardPage()
    page.create()
    return page
