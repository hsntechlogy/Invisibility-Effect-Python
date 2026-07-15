import cv2
import numpy as np
import mss
import time
import pygetwindow as gw  # Added this to control the windows
from tensorflow.keras.models import load_model

# 1. Load your trained model
# Make sure the filename matches your actual model file!
model = load_model('content_model.h5') 

def process_screen():
    # Fixed the deprecation warning by using capital MSS()
    with mss.MSS() as sct:
        # Get the primary monitor
        monitor = sct.monitors[1]
        
        print("Starting screen monitor... Press Ctrl+C to stop.")
        
        while True:
            # Capture the screen
            screenshot = sct.grab(monitor)
            
            # Convert the raw capture to a numpy array
            img_bgra = np.array(screenshot)
            
            # --- CRITICAL FIXES START HERE ---
            # 1. Convert BGRA (mss format) to RGB (AI model format)
            img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
            
            # 2. Resize to 224x224 (Standard size for MobileNetV2 and most image models)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # 3. Normalize the image (scale pixel values from 0-255 down to 0.0-1.0)
            img_normalized = img_resized / 255.0
            
            # 4. Add the "batch" dimension. Model expects (1, 224, 224, 3) not just (224, 224, 3)
            img_input = np.expand_dims(img_normalized, axis=0)
            # --- CRITICAL FIXES END HERE ---
            
            # Run the prediction
            prediction = model.predict(img_input, verbose=0)
            
            # Assuming your model outputs exactly two categories: [Safe, Prohibited]
            safe_prob = prediction[0][0]
            prohibited_prob = prediction[0][1]
            
            print(f"Seeing Browser -> Safe: {safe_prob:.2f} | Prohibited: {prohibited_prob:.2f}")
            
            # The trigger logic
            if prohibited_prob > 0.85:
                print("🚫 Prohibited content detected! Minimizing...")
                
                # --- ACTUAL MINIMIZATION LOGIC ---
                try:
                    # Look for Chrome first
                    windows = gw.getWindowsWithTitle('Chrome')
                    if windows:
                        # Minimize the Chrome window
                        windows[0].minimize()
                        print("Chrome window minimized successfully.")
                    else:
                        # Fallback: If Chrome isn't found, minimize the currently active window
                        active_window = gw.getActiveWindow()
                        if active_window:
                            active_window.minimize()
                            print("Active window minimized.")
                except Exception as e:
                    print(f"Error trying to minimize window: {e}")
                # ---------------------------------
                
                # Pause for 3 seconds so it doesn't try to minimize 100 times while the animation plays
                time.sleep(3) 
            
            # Small sleep to prevent your CPU from maxing out
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        process_screen()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.") #Ctrl+C