import mss#handles screen capturing.
import cv2 #handles image processing and displays the video window.
import numpy as np
import os
import pygetwindow as gw

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROHIBITED_DIR = os.path.join(BASE_DIR, "data", "prohibited")
SAFE_DIR = os.path.join(BASE_DIR, "data", "safe")
os.makedirs(PROHIBITED_DIR, exist_ok=True)
os.makedirs(SAFE_DIR, exist_ok=True)

def get_browser_region():
    # Replace 'Chrome' with your browser name if you change browsers
    windows = gw.getWindowsWithTitle('Chrome') 
    if windows:
        win = windows[0]
        return {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
    return None

count = 0
# Fixed the DeprecationWarning with capital MSS()
with mss.MSS() as sct:
    print("Data Collector Running... Select the OpenCV window to use keys.")
    print("[a] -> Save SAFE | [s] -> Save PROHIBITED | [q] -> Quit")
    
    while True:
        region = get_browser_region()
        if not region:
            continue
            
        screenshot = sct.grab(region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Show what the AI sees
        cv2.imshow("Data Collection - Active Browser", cv2.resize(frame, (640, 480)))
        
        # Make sure the 'Data Collection' window is clicked/active when pressing keys!
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('s'): # Prohibited
            cv2.imwrite(os.path.join(PROHIBITED_DIR, f"img_{count}.jpg"), frame)
            print(f"🚫 PROHIBITED saved: img_{count}.jpg")
            count += 1
        elif key == ord('a'): # Safe
            cv2.imwrite(os.path.join(SAFE_DIR, f"img_{count}.jpg"), frame)
            print(f"✅ SAFE saved: img_{count}.jpg")
            count += 1

cv2.destroyAllWindows()


 