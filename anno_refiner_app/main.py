#!/usr/bin/env python3
"""
Annotation Refiner - Main Entry Point

A web-based tool for detecting and correcting YOLO annotation issues
using Cleanlab's object detection analysis.

Usage:
    python main.py [--host HOST] [--port PORT]
"""
import argparse
import logging
from pathlib import Path

from nicegui import ui, app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_routes():
    """Setup application routes"""
    
    @ui.page('/')
    def dashboard_page():
        """Dashboard page"""
        from src.ui.page_dashboard import create_dashboard
        create_dashboard()
    
    @ui.page('/annotator')
    def annotator_page():
        """Annotator page (placeholder for now)"""
        from src.state import app_state
        
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):
            ui.label('Annotation Tool').classes('text-2xl font-bold')
            ui.label('(Coming in Stage 5-6)').classes('text-gray-500')
            
            ui.separator()
            
            # Show queue info
            queue_size = len(app_state.annotation_queue)
            current_idx = app_state.current_annotation_index
            
            if queue_size > 0:
                ui.label(f'Annotation Queue: {queue_size} samples').classes('mt-4')
                ui.label(f'Current: {current_idx + 1} / {queue_size}')
                
                # Show current item
                if current_idx < queue_size:
                    item = app_state.annotation_queue[current_idx]
                    with ui.card().classes('w-full mt-4 p-4'):
                        ui.label(f'Image: {item.image_path}')
                        ui.label(f'Issue Type: {item.issue_type.value}')
                        ui.label(f'Score: {item.score:.4f}')
                        ui.label(f'Box Index: {item.box_index}')
                        
                        # Show image preview
                        img_path = Path(app_state.config.images_path) / item.image_path
                        if img_path.exists():
                            ui.image(str(img_path)).classes('max-w-xl mt-4')
            else:
                ui.label('No samples in queue').classes('mt-4 text-gray-500')
            
            # Navigation
            with ui.row().classes('mt-4 gap-4'):
                ui.button('Back to Dashboard', on_click=lambda: ui.navigate.to('/')).classes('bg-gray-500 text-white')
                
                if queue_size > 0:
                    def prev_sample():
                        if app_state.current_annotation_index > 0:
                            app_state.current_annotation_index -= 1
                            ui.navigate.to('/annotator')
                    
                    def next_sample():
                        if app_state.current_annotation_index < queue_size - 1:
                            app_state.current_annotation_index += 1
                            ui.navigate.to('/annotator')
                    
                    ui.button('Previous', on_click=prev_sample).props('outline')
                    ui.button('Next', on_click=next_sample).props('outline')


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Annotation Refiner')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    
    logger.info(f'Starting Annotation Refiner on {args.host}:{args.port}')
    
    # Setup routes
    setup_routes()
    
    # Run the app
    ui.run(
        host=args.host,
        port=args.port,
        title='Annotation Refiner',
        reload=args.reload,
        show=False  # Don't auto-open browser
    )


if __name__ == '__main__':
    main()
