# create_icon.py
"""Create simple icons for the application"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_icon():
    """Create a simple icon with 'QCI' text"""
    size = 256
    img = Image.new('RGB', (size, size), color='#2c3e50')
    draw = ImageDraw.Draw(img)
    
    # Draw a circle
    draw.ellipse([20, 20, size-20, size-20], fill='#3498db', outline='white', width=5)
    
    # Add text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 100)
    except:
        font = ImageFont.load_default()
    
    draw.text((size//2, size//2), "QCI", fill='white', anchor="mm", font=font)
    
    # Save as PNG
    img.save("icon.png")
    
    # Convert to ICO for Windows
    img.save("icon.ico", format="ICO", sizes=[(256, 256)])
    
    # For macOS, you'd need icns format (use iconutil on Mac)
    print("Created: icon.png, icon.ico")
    
    if os.name == 'posix' and os.uname().sysname == 'Darwin':
        print("Note: For macOS, convert to .icns using:")
        print("  mkdir icon.iconset")
        print("  sips -z 16 16 icon.png --out icon.iconset/icon_16x16.png")
        print("  ... (various sizes)")
        print("  iconutil -c icns icon.iconset")

if __name__ == "__main__":
    create_simple_icon()
