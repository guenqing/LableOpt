"""
Dashboard page - Analysis control panel with visualization
"""
import asyncio
import tempfile
import os
import time
from pathlib import Path
from nicegui import ui, app
# matplotlib and PIL imports are moved inside _generate_visualization to avoid DLL loading issues

from ..state import app_state
from ..models import IssueType, IssueItem, ClassMapping
from ..core.file_manager import (
    backup_folder,
    validate_output_path,
    parse_data_for_dashboard,
)
from ..core.path_utils import resolve_with_base_dir
from ..core.analyzer import CleanlabAnalyzer
from ..core.label_utils import build_class_name_to_id, get_image_size, read_label_file, resolve_label_path


class DashboardPage:
    """Dashboard page for configuring and running Cleanlab analysis"""
    
    def __init__(self):
        self.progress_bar = None
        self.progress_label = None
        self.parse_button = None
        self.run_button = None
        self.goto_button = None
        self._analysis_progress_timer = None

        # Path parsing progress
        self.parse_progress_bar = None
        self.parse_progress_label = None
        self._parse_task = None
        
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
        self.human_status = None

        # Pending analysis count
        self.pending_count = None
        self.pending_detail = None

        # Count labels under inputs
        self.images_count_label = None
        self.gt_count_label = None
        self.pred_count_label = None
        self.output_count_label = None
        self.human_count_label = None

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

                ui.label(f'Base Dir: {app_state.config.base_dir}').classes('text-[10px] text-gray-400 break-all')
                
                # Images path
                ui.label('Images Path').classes('text-xs font-medium text-gray-500 mt-1')
                with ui.row().classes('w-full items-center gap-1'):
                    self.images_input = ui.input().classes('flex-grow').props('dense outlined size=small placeholder="relative to base dir"')
                    self.images_status = ui.label('').classes('text-xs whitespace-nowrap')
                self.images_count_label = ui.label('').classes('text-[10px] text-gray-400')
                
                # GT Labels path
                ui.label('GT Labels Path (opt)').classes('text-xs font-medium text-gray-500')
                with ui.row().classes('w-full items-center gap-1'):
                    self.gt_input = ui.input().classes('flex-grow').props('dense outlined size=small placeholder="relative to base dir"')
                    self.gt_status = ui.label('').classes('text-xs whitespace-nowrap')
                self.gt_count_label = ui.label('').classes('text-[10px] text-gray-400')
                
                # Pred Labels path
                ui.label('Pred Labels Path (opt)').classes('text-xs font-medium text-gray-500')
                with ui.row().classes('w-full items-center gap-1'):
                    self.pred_input = ui.input().classes('flex-grow').props('dense outlined size=small placeholder="relative to base dir"')
                    self.pred_status = ui.label('').classes('text-xs whitespace-nowrap')
                self.pred_count_label = ui.label('').classes('text-[10px] text-gray-400')

                # Parse progress (for path validation)
                self.parse_progress_label = ui.label('').classes('text-[10px] text-gray-400')
                self.parse_progress_bar = ui.linear_progress().classes('w-full')
                self.parse_progress_bar.props('indeterminate')
                self.parse_progress_bar.visible = False
                
                # Output Path
                ui.label('Output Path').classes('text-xs font-medium text-gray-500')
                with ui.row().classes('w-full items-center gap-1'):
                    self.output_input = ui.input().classes('flex-grow').props('dense outlined size=small placeholder="relative to base dir"')
                    self.output_status = ui.label('').classes('text-xs whitespace-nowrap')
                self.output_count_label = ui.label('').classes('text-[10px] text-gray-400')
                
                # Human Verified Annotation Path
                ui.label('Human Verified Annotation Path (opt)').classes('text-xs font-medium text-gray-500')
                with ui.row().classes('w-full items-center gap-1'):
                    self.human_verified_input = ui.input().classes('flex-grow').props('dense outlined size=small placeholder="relative to base dir"')
                    self.human_status = ui.label('').classes('text-xs whitespace-nowrap')
                self.human_count_label = ui.label('').classes('text-[10px] text-gray-400')
                
                # Classes file
                ui.label('Classes File (opt)').classes('text-xs font-medium text-gray-500')
                self.classes_input = ui.input().classes('w-full').props('dense outlined size=small')

                # Pending analysis samples
                with ui.row().classes('w-full items-center justify-between mt-1'):
                    ui.label('Pending Samples').classes('text-xs font-medium text-gray-500')
                    self.pending_count = ui.label('--').classes('text-lg font-bold text-indigo-700')
                self.pending_detail = ui.label('').classes('text-[10px] text-gray-400')
                
                ui.separator().classes('my-2')
                
                # Parse data button
                self.parse_button = ui.button(
                    'Parse Data',
                    on_click=self._on_parse_data,
                    icon='auto_fix_high'
                ).classes('w-full').props('color=negative').tooltip('手动触发路径解析与统计')

                # Run button
                self.run_button = ui.button(
                    'RUN ANALYSIS',
                    on_click=self._run_analysis,
                    icon='play_arrow'
                ).classes('w-full').props('color=primary').tooltip('运行分析（需设置Images、GT、Pred路径）')
                
                # Progress
                self.progress_label = ui.label('').classes('text-xs text-gray-500')
                self.progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                self.progress_bar.visible = False

                # Keep UI progress synced from global state (safe across reconnects)
                self._analysis_progress_timer = ui.timer(0.2, self._refresh_analysis_progress)
                
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
        
        # Initialize paths in config (as absolute paths)
        self._sync_paths_to_config()
    
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
                self.refresh_button = ui.button(icon='refresh', on_click=self._on_refresh_topk).props('flat dense size=sm').tooltip('刷新TopK结果')
                self.refresh_button.disable()
            
            # Go to annotation button
            with ui.row().classes('justify-end mt-2'):
                self.goto_button = ui.button(
                    'Go to Annotation Tool',
                    on_click=self._goto_annotation,
                    icon='edit'
                ).props('color=positive').tooltip('进入标注工具（分析模式或直接标注模式）')
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
        self._sync_paths_to_config()
        self._clear_parse_results()
    
    def _on_output_path_change(self, e=None):
        """Handle output path input change"""
        self._sync_paths_to_config()
        self._validate_output_path()
        self._clear_parse_results()

    def _to_abs(self, user_value: str) -> str:
        return resolve_with_base_dir(app_state.config.base_dir, user_value or '')

    def _sync_paths_to_config(self) -> None:
        """Sync UI input (relative/absolute) into config as absolute paths."""
        app_state.config.images_path = self._to_abs(self.images_input.value)
        app_state.config.gt_labels_path = self._to_abs(self.gt_input.value)
        app_state.config.pred_labels_path = self._to_abs(self.pred_input.value)
        app_state.config.output_path = self._to_abs(self.output_input.value)
        app_state.config.human_verified_path = self._to_abs(self.human_verified_input.value)

    def _clear_parse_results(self) -> None:
        """Clear parse result displays after path changes."""
        if self.images_status:
            self.images_status.text = ''
        if self.gt_status:
            self.gt_status.text = ''
        if self.pred_status:
            self.pred_status.text = ''
        if self.human_status:
            self.human_status.text = ''
        if self.images_count_label:
            self.images_count_label.text = ''
        if self.gt_count_label:
            self.gt_count_label.text = ''
        if self.pred_count_label:
            self.pred_count_label.text = ''
        if self.output_count_label:
            self.output_count_label.text = ''
        if self.human_count_label:
            self.human_count_label.text = ''
        if self.pending_count:
            self.pending_count.text = '--'
        if self.pending_detail:
            self.pending_detail.text = ''
        if self.parse_progress_bar:
            self.parse_progress_bar.visible = False
        if self.parse_progress_label:
            self.parse_progress_label.text = ''

    def _on_parse_data(self) -> None:
        if self._parse_task and not self._parse_task.done():
            self._parse_task.cancel()
        self._parse_task = asyncio.create_task(self._parse_data_async())

    async def _parse_data_async(self) -> None:
        """Parse dashboard data on demand and update status labels."""

        images_path = (app_state.config.images_path or '').strip()
        gt_path = (app_state.config.gt_labels_path or '').strip()
        pred_path = (app_state.config.pred_labels_path or '').strip()
        output_path = (app_state.config.output_path or '').strip()
        human_path = (app_state.config.human_verified_path or '').strip()
        if not any([images_path, gt_path, pred_path, output_path, human_path]):
            self._clear_parse_results()
            return

        if self.parse_progress_label:
            self.parse_progress_label.text = 'Parsing data...'
        if self.parse_progress_bar:
            self.parse_progress_bar.visible = True

        start_time = time.time()
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                lambda: parse_data_for_dashboard(
                    images_path=images_path,
                    gt_labels_path=gt_path,
                    pred_labels_path=pred_path,
                    output_path=output_path,
                    human_verified_path=human_path,
                ),
            )

            if self.images_status:
                if not images_path:
                    self.images_status.text = ''
                elif stats.get('images_exists'):
                    self.images_status.text = 'OK'
                    self.images_status.classes(remove='text-red-500', add='text-green-600')
                else:
                    self.images_status.text = 'Not found'
                    self.images_status.classes(remove='text-green-600', add='text-red-500')
            if self.images_count_label:
                if stats.get('images_count') is not None:
                    self.images_count_label.text = f'{int(stats["images_count"])} imgs'
                else:
                    self.images_count_label.text = ''

            if self.gt_status:
                if not gt_path:
                    self.gt_status.text = ''
                elif Path(gt_path).exists():
                    self.gt_status.text = 'OK'
                    self.gt_status.classes(remove='text-red-500', add='text-green-600')
                else:
                    self.gt_status.text = 'Not found'
                    self.gt_status.classes(remove='text-green-600', add='text-red-500')
            if self.gt_count_label:
                if gt_path:
                    self.gt_count_label.text = (
                        f'valid: {int(stats.get("gt_valid") or 0)} '
                        f'missing_img: {int(stats.get("gt_missing_img") or 0)}'
                    )
                else:
                    self.gt_count_label.text = ''

            if self.pred_status:
                if not pred_path:
                    self.pred_status.text = ''
                elif Path(pred_path).exists():
                    self.pred_status.text = 'OK'
                    self.pred_status.classes(remove='text-red-500', add='text-green-600')
                else:
                    self.pred_status.text = 'Not found'
                    self.pred_status.classes(remove='text-green-600', add='text-red-500')
            if self.pred_count_label:
                if pred_path:
                    self.pred_count_label.text = (
                        f'valid: {int(stats.get("pred_valid") or 0)} '
                        f'missing_img: {int(stats.get("pred_missing_img") or 0)}'
                    )
                else:
                    self.pred_count_label.text = ''

            if self.output_count_label:
                if output_path:
                    self.output_count_label.text = (
                        f'valid: {int(stats.get("output_valid") or 0)} '
                        f'missing_img: {int(stats.get("output_missing_img") or 0)}'
                    )
                else:
                    self.output_count_label.text = ''

            if self.human_status:
                if not human_path:
                    self.human_status.text = ''
                elif Path(human_path).exists():
                    self.human_status.text = 'OK'
                    self.human_status.classes(remove='text-red-500', add='text-green-600')
                else:
                    self.human_status.text = 'Not found'
                    self.human_status.classes(remove='text-green-600', add='text-red-500')
            if self.human_count_label:
                if human_path:
                    self.human_count_label.text = (
                        f'valid: {int(stats.get("human_valid") or 0)} '
                        f'missing_img: {int(stats.get("human_missing_img") or 0)}'
                    )
                else:
                    self.human_count_label.text = ''

            pending = stats.get('pending')
            if self.pending_count:
                self.pending_count.text = '--' if pending is None else str(int(pending))
            if self.pending_detail:
                if pending is None and not images_path:
                    self.pending_detail.text = 'Fill Images to compute'
                else:
                    self.pending_detail.text = ''

            elapsed = time.time() - start_time
            import logging
            logging.getLogger(__name__).info(
                f'Path parsing done in {elapsed:.3f}s: '
                f'imgs={stats.get("images_count")} gt_valid={stats.get("gt_valid")} pred_valid={stats.get("pred_valid")} '
                f'out_valid={stats.get("output_valid")} human_valid={stats.get("human_valid")} '
                f'pending={stats.get("pending")} mode={stats.get("mode")}'
            )
        except asyncio.CancelledError:
            return
        except Exception as ex:
            import logging
            logging.getLogger(__name__).error(f'Path parsing failed: {ex}')
        finally:
            if self.parse_progress_bar:
                self.parse_progress_bar.visible = False
            if self.parse_progress_label:
                self.parse_progress_label.text = ''

    def _validate_output_path(self):
        """Validate output path and check for conflicts"""
        output_path = app_state.config.output_path or ''
        gt_path = app_state.config.gt_labels_path or ''
        pred_path = app_state.config.pred_labels_path or ''
        
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
            self._safe_notify(
                'Output Path is required. Please set Output Path before running analysis.',
                type='negative',
                timeout=5000,
            )
            return

        # Lightweight validation (avoid scanning huge directories on click)
        images_path = (app_state.config.images_path or '').strip()
        gt_path = (app_state.config.gt_labels_path or '').strip()
        pred_path = (app_state.config.pred_labels_path or '').strip()

        # Enforce: analysis requires Images + GT + Pred
        if not images_path or not gt_path or not pred_path:
            self._safe_notify(
                'RUN ANALYSIS requires Images Path, GT Labels Path, and Pred Labels Path.',
                type='warning',
                timeout=5000,
            )
            return

        if not Path(images_path).exists():
            self._safe_notify('Images Path not found. Please check the path.', type='negative')
            return
        if not Path(gt_path).exists():
            self._safe_notify('GT Labels Path not found. Please check the path.', type='negative')
            return
        if not Path(pred_path).exists():
            self._safe_notify('Pred Labels Path not found. Please check the path.', type='negative')
            return
        
        # Record start time
        start_time = time.time()
        
        app_state.reset_analysis()
        app_state.is_analyzing = True
        self._update_progress('Starting analysis...', 0.0)

        try:
            if self.run_button:
                self.run_button.disable()
            if self.progress_bar:
                self.progress_bar.visible = True
        except RuntimeError:
            # client disconnected; keep analysis running via global state
            pass
        
        try:
            self._update_progress('Preparing output directory...', 0.01)
            Path(app_state.config.output_path).mkdir(parents=True, exist_ok=True)
            
            # Backup only if checkbox is checked
            if self.backup_checkbox.value:
                self._update_progress('Backing up GT folder...', 0.02)
                try:
                    loop = asyncio.get_event_loop()
                    backup_path = await loop.run_in_executor(None, lambda: backup_folder(app_state.config.gt_labels_path))
                    self._safe_notify(f'Backup created: {Path(backup_path).name}', type='positive')
                except Exception as ex:
                    self._safe_notify(f'Backup failed: {ex}', type='warning')
            
            analyzer = CleanlabAnalyzer(
                images_path=app_state.config.images_path,
                pred_labels_path=app_state.config.pred_labels_path,
                gt_labels_path=app_state.config.gt_labels_path,
                output_path=app_state.config.output_path,
                human_verified_path=app_state.config.human_verified_path,
                progress_callback=self._update_progress,
                class_mapping=app_state.class_mapping,
            )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, analyzer.prepare_data)
            
            results = await loop.run_in_executor(None, lambda: analyzer.analyze(top_k=1000))
            
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
            try:
                self._update_results_display()
                if self.goto_button:
                    self.goto_button.enable()
                if self.refresh_button:
                    self.refresh_button.enable()
            except RuntimeError:
                # client disconnected
                pass
            
            # Show counts for each category (these are stored with top_k=1000 from cleanlab)
            n_overlooked = len(app_state.results.overlooked)
            n_swapped = len(app_state.results.swapped)
            n_badloc = len(app_state.results.bad_located)
            self._safe_notify(
                f'Analysis complete: {n_overlooked} overlooked, {n_swapped} swapped, {n_badloc} bad located',
                type='positive',
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
            
            self._update_progress(f'Analysis failed: {ex}', 1.0)
            self._safe_notify(f'Analysis failed: {ex}', type='negative')
            import traceback
            traceback.print_exc()
        finally:
            app_state.is_analyzing = False
            try:
                if self.run_button:
                    self.run_button.enable()
            except RuntimeError:
                pass
    
    def _update_progress(self, message: str, percentage: float):
        """Update global progress state (safe to call from worker threads)."""
        app_state.analysis_message = message or ''
        try:
            pct = float(percentage)
        except Exception:
            pct = 0.0
        app_state.analysis_progress = max(0.0, min(1.0, pct))

    def _refresh_analysis_progress(self) -> None:
        """Refresh UI progress widgets from global state."""
        try:
            if self.progress_label:
                self.progress_label.text = app_state.analysis_message or ''
            if self.progress_bar:
                self.progress_bar.value = float(app_state.analysis_progress or 0.0)
                self.progress_bar.visible = bool(app_state.is_analyzing)

            # keep button state sensible after reconnects
            if self.run_button:
                if app_state.is_analyzing:
                    self.run_button.disable()
                else:
                    self.run_button.enable()
            if self.goto_button:
                if app_state.is_analyzing:
                    self.goto_button.disable()
                elif app_state.analysis_complete:
                    self.goto_button.enable()
                else:
                    images_path = (app_state.config.images_path or '').strip()
                    output_path = (app_state.config.output_path or '').strip()
                    if images_path and Path(images_path).exists() and output_path:
                        self.goto_button.enable()
                    else:
                        self.goto_button.disable()
            if self.refresh_button:
                if app_state.analysis_complete and (not app_state.is_analyzing):
                    self.refresh_button.enable()
                else:
                    self.refresh_button.disable()
        except RuntimeError:
            # client disconnected
            return

    def _safe_notify(self, message: str, type: str = 'info', timeout: int = 3000) -> None:
        """Notify if a client is available; otherwise just ignore."""
        try:
            ui.notify(message, type=type, timeout=timeout)
        except RuntimeError:
            return
    
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
        # Import matplotlib here to avoid DLL loading issues on module import
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        images_dir = Path(app_state.config.images_path)
        gt_dir = Path(app_state.config.gt_labels_path)
        pred_dir = Path(app_state.config.pred_labels_path)
        
        rel_path = Path(item.image_path)
        img_path = images_dir / rel_path
        key = rel_path.with_suffix('')
        gt_label_path = resolve_label_path(gt_dir, key)
        pred_label_path = resolve_label_path(pred_dir, key)
        class_name_to_id = build_class_name_to_id(app_state.class_mapping)
        
        # Load image using matplotlib's imread to avoid PIL dependency
        img = plt.imread(img_path)
        img_h, img_w, _ = img.shape
        
        # Load boxes
        gt_boxes = read_label_file(gt_label_path, img_w, img_h, has_confidence=False, class_name_to_id=class_name_to_id)
        pred_boxes = read_label_file(pred_label_path, img_w, img_h, has_confidence=True, class_name_to_id=class_name_to_id)
        
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
        # Always require Images Path + Output Path (even when skipping analysis)
        images_path = (app_state.config.images_path or '').strip()
        output_path = (app_state.config.output_path or '').strip()
        if not images_path:
            ui.notify('Images Path is required.', type='warning')
            return
        if not Path(images_path).exists():
            ui.notify('Images Path not found. Please check the path.', type='negative')
            return
        if not output_path:
            ui.notify('Output Path is required.', type='warning')
            return

        # Ensure output root exists for saving tmp files
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
        except Exception as ex:
            ui.notify(f'Failed to create Output Path: {ex}', type='negative')
            return

        # If analysis has been run, keep the original behavior (issue-based queue)
        if app_state.analysis_complete:
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
            ui.notify(
                f'Starting with {len(app_state.annotation_queue)} samples from {type_str} (TopK={top_k})',
                type='info'
            )
            ui.navigate.to('/annotator')
            return

        # Otherwise, allow direct annotation without cleanlab analysis.
        gt_path = (app_state.config.gt_labels_path or '').strip()
        pred_path = (app_state.config.pred_labels_path or '').strip()
        human_verified_path = (app_state.config.human_verified_path or '').strip()

        if gt_path and not Path(gt_path).exists():
            ui.notify('GT Labels Path not found. Please check the path.', type='negative')
            return
        if pred_path and not Path(pred_path).exists():
            ui.notify('Pred Labels Path not found. Please check the path.', type='negative')
            return

        if human_verified_path and not Path(human_verified_path).exists():
            ui.notify('Human Verified Path not found; it will be ignored.', type='warning')
            human_verified_path = ''

        try:
            from ..core.file_manager import collect_annotation_image_paths
            rels = collect_annotation_image_paths(
                images_path=images_path,
                gt_labels_path=gt_path,
                pred_labels_path=pred_path,
                output_path=output_path,
                human_verified_path=human_verified_path,
            )
        except Exception as ex:
            ui.notify(f'Failed to build annotation queue: {ex}', type='negative', timeout=5000)
            return

        if not rels:
            ui.notify('No samples found after applying intersection and skip rules.', type='warning')
            return

        from ..models import IssueItem, IssueType
        app_state.annotation_queue = [
            IssueItem(
                image_path=p,
                issue_type=IssueType.DIRECT,
                score=1.0,
                box_index=None,
            )
            for p in rels
        ]
        app_state.current_annotation_index = 0
        ui.notify(f'Starting annotation with {len(app_state.annotation_queue)} samples', type='info')
        ui.navigate.to('/annotator')


def create_dashboard():
    """Create dashboard page"""
    page = DashboardPage()
    page.create()
    return page
