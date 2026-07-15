"""import mss
import cv2
import numpy as np

# Monitor region: You can adjust this to a smaller area if 800x600 is still lagging
monitor = {"top": 0, "left": 0, "width": 800, "height": 600}

print("Starting screen capture. Press 'q' to stop.")

try:
    with mss.mss() as sct:
        while True:
            # 1. Capture the screen data
            img = sct.grab(monitor)
            
            # 2. Convert to numpy array
            frame = np.array(img)
            
            # 3. Convert colors
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 4. Resize the frame for display (Downscale to 50% for stability)
            display_frame = cv2.resize(frame, (400, 300))
            
            # 5. Display the downscaled frame
            cv2.imshow("Screen Monitor (Downscaled)", display_frame)
            
            # 6. Wait for key press (25ms is standard)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cv2.destroyAllWindows()
    print("Capture stopped.") """

 # 2nd 

"""
import mss
import cv2
import numpy as np
import os

# Get the directory where your script is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths
PROHIBITED_DIR = os.path.join(BASE_DIR, "data", "prohibited")
SAFE_DIR = os.path.join(BASE_DIR, "data", "safe")

# Create folders
os.makedirs(PROHIBITED_DIR, exist_ok=True)
os.makedirs(SAFE_DIR, exist_ok=True)

count = 0
monitor = {"top": 0, "left": 0, "width": 800, "height": 600}

with mss.mss() as sct:
    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        cv2.imshow("Data Collection", frame)
        
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Use the absolute path here
            path = os.path.join(PROHIBITED_DIR, f"img_{count}.jpg")
            cv2.imwrite(path, frame)
            print(f"Saved to: {path}")
            count += 1
        elif key == ord('a'):
            path = os.path.join(SAFE_DIR, f"img_{count}.jpg")
            cv2.imwrite(path, frame)
            print(f"Saved to: {path}")
            count += 1

cv2.destroyAllWindows() 

"""
"""
import mss
import cv2
import numpy as np

# A simple detector function (Placeholder for your future AI)
def is_prohibited(frame):
    # Calculate the mean brightness of the frame
    # A simple metric: if the frame is 'bright' or has high variance, 
    # flag it for demonstration. 
    # You will replace this logic with your AI model later!
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 150:  # Threshold: change this to test triggers
        return True
    return False

monitor = {"top": 0, "left": 0, "width": 800, "height": 600}

with mss.mss() as sct:
    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # --- THE BRAIN ---
        if is_prohibited(frame):
            print("ALERT: Prohibited content detected!")
            # In Phase Three, we add the "Hand" (e.g., cv2.destroyAllWindows() or minimize)
        # -----------------
        
        cv2.imshow("Monitoring...", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()"""
    