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
        """Annotator page"""
        from src.ui.page_annotator import create_annotator
        create_annotator()


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
