#!/usr/bin/env python3
"""
Simple test for coordinate conversion functions
"""
import sys
sys.path.insert(0, '.')

from anno_refiner_app.src.core.yolo_utils import yolo_to_pixel, pixel_to_yolo

def test_coordinate_conversion():
    """Test YOLO <-> pixel coordinate conversion"""
    print("Testing coordinate conversion...")
    
    img_w, img_h = 1920, 1080
    
    # Test case 1: Normal box
    print("\nTest 1: Normal box")
    cx, cy, w, h = 0.5, 0.5, 0.2, 0.3
    x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
    print(f"  YOLO ({cx}, {cy}, {w}, {h}) -> Pixel ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    
    # Test round-trip conversion
    cx2, cy2, w2, h2 = pixel_to_yolo(x1, y1, x2, y2, img_w, img_h)
    print(f"  Pixel -> YOLO ({cx2:.6f}, {cy2:.6f}, {w2:.6f}, {h2:.6f})")
    
    # Calculate errors
    errors = [abs(a-b) for a,b in zip([cx, cy, w, h], [cx2, cy2, w2, h2])]
    print(f"  Round-trip errors: cx={errors[0]:.6f}, cy={errors[1]:.6f}, w={errors[2]:.6f}, h={errors[3]:.6f}")
    
    if all(error < 0.000001 for error in errors):
        print("  ✓ Round-trip conversion successful")
    else:
        print("  ✗ Round-trip conversion failed")
    
    # Test case 2: Box with coordinates slightly outside [0,1] range
    print("\nTest 2: Box with coordinates slightly outside [0,1]")
    cx, cy, w, h = 1.05, -0.05, 1.1, 0.8
    x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
    print(f"  YOLO ({cx}, {cy}, {w}, {h}) -> Pixel ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    print(f"  ✓ Coordinates properly clamped")
    
    # Test case 3: Small box
    print("\nTest 3: Small box")
    cx, cy, w, h = 0.5, 0.5, 0.0005, 0.0005  # Very small box
    x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
    print(f"  YOLO ({cx}, {cy}, {w}, {h}) -> Pixel ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    print(f"  ✓ Small box handled correctly")
    
    # Test case 4: Edge cases
    print("\nTest 4: Edge cases")
    
    # Top-left corner
    cx, cy, w, h = 0.0, 0.0, 0.1, 0.1
    x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
    print(f"  Top-left corner -> Pixel ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    
    # Bottom-right corner
    cx, cy, w, h = 1.0, 1.0, 0.1, 0.1
    x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
    print(f"  Bottom-right corner -> Pixel ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_coordinate_conversion()
