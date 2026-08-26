import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

def main():
    # ---------------------------------------------------------
    # 1. INITIALIZE MEDIAPIPE TASKS API
    # ---------------------------------------------------------
    hand_model_path = r'D:\pythonseries\filter\hand_landmarker.task'
    base_options_hands = python.BaseOptions(model_asset_path=hand_model_path)
    options_hands = vision.HandLandmarkerOptions(base_options=base_options_hands, num_hands=1)
    hand_landmarker = vision.HandLandmarker.create_from_options(options_hands)
    
    seg_model_path = r'D:\pythonseries\filter\selfie_segmenter_landscape.tflite'
    base_options_seg = python.BaseOptions(model_asset_path=seg_model_path)
    options_seg = vision.ImageSegmenterOptions(
        base_options=base_options_seg, 
        output_category_mask=True
    )
    segmenter = vision.ImageSegmenter.create_from_options(options_seg)
    
    # ---------------------------------------------------------
    # 2. CAMERA SETUP & BACKGROUND CAPTURE
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(0)
    window_name = "Liquid Water Silhouette Effect"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("STEP OUT OF THE FRAME! Capturing clean background...")
    start_time = time.time()
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.ascontiguousarray(np.flip(frame, axis=1))
        remaining = int(3 - (time.time() - start_time))
        
        display_frame = frame.copy()
        cv2.putText(display_frame, f"STEP OUT OF FRAME! Capturing in: {remaining}s", (40, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    background_frames = []
    for i in range(25):
        ret, bg_frame = cap.read()
        if ret:
            background_frames.append(np.flip(bg_frame, axis=1))
    
    background = np.median(background_frames, axis=0).astype(np.uint8) if background_frames else np.flip(cap.read()[1], axis=1)
    print("Background secured! Step back in.")

    prev_mask = None

    # ---------------------------------------------------------
    # 3. MAIN LOOP
    # ---------------------------------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.ascontiguousarray(np.flip(frame, axis=1))
        h, w, c = frame.shape
        
        # Continuously adapt background to match minor lighting changes
        background = cv2.addWeighted(background, 0.995, frame, 0.005, 0)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        hand_results = hand_landmarker.detect(mp_image)
        effect_active = bool(hand_results.hand_landmarks)

        if effect_active:
            seg_results = segmenter.segment(mp_image)
            mask = seg_results.category_mask.numpy_view()
            mask = np.squeeze(mask)
            
            if mask.max() <= 1:
                bg_sample = (mask[0, 0] + mask[0, -1] + mask[-1, 0] + mask[-1, -1]) / 4.0
                alpha_mask = (1.0 - mask.astype(np.float32)) if bg_sample > 0.5 else mask.astype(np.float32)
            else:
                alpha_mask = mask.astype(np.float32) / 255.0
                if alpha_mask[0, 0] > 0.5:
                    alpha_mask = 1.0 - alpha_mask

            # Smooth response for fast hand and head tracking
            if prev_mask is not None:
                alpha_mask = cv2.addWeighted(alpha_mask, 0.5, prev_mask, 0.5, 0)
            prev_mask = alpha_mask.copy()

            # Ensure complete coverage across hands, face, and head
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            alpha_mask_u8 = (alpha_mask * 255).astype(np.uint8)
            alpha_mask_u8 = cv2.morphologyEx(alpha_mask_u8, cv2.MORPH_CLOSE, kernel)
            
            # Soft liquid edge blending (removes harsh solid cutout shapes)
            alpha_mask = cv2.GaussianBlur(alpha_mask_u8.astype(np.float32) / 255.0, (23, 23), 0)
            condition = np.stack((alpha_mask,) * 3, axis=-1)
            
            # --- HEAVY LIQUID WATER REFRACTION & INTERNAL FEATURES WARP ---
            # Using the live camera frame itself as the texture map so your facial features, eyes, and skin 
            # get warped inside like a transparent liquid body lens, rather than just showing a flat background.
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smooth_gray = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            
            grad_x = cv2.Sobel(smooth_gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(smooth_gray, cv2.CV_32F, 0, 1, ksize=3)
            
            y_indices, x_indices = np.indices((h, w), dtype=np.float32)
            
            # Deep liquid wave distortion map
            map_x = np.clip(x_indices + grad_x * 0.8, 0, w - 1)
            map_y = np.clip(y_indices + grad_y * 0.8, 0, h - 1)
            
            # Remap live frame so skin/features are heavily refracted like flowing water
            liquid_features = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            # Blend refracted features with background to achieve a true see-through water look
            water_lens = cv2.addWeighted(liquid_features, 0.45, background, 0.55, 0)
            
            # Add subtle water shimmer highlight sheen
            water_lens = cv2.addWeighted(water_lens, 0.9, np.full_like(water_lens, 255), 0.1, 0)

            # Seamlessly blend the liquid see-through shape over the live scene
            final_output = (condition * water_lens + (1 - condition) * frame).astype(np.uint8)
        else:
            final_output = frame

        cv2.imshow(window_name, final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hand_landmarker.close()
    segmenter.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 