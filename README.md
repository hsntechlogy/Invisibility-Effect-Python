# AI Screen-Tracking Content Blocker

An advanced Python-based computer vision pipeline that monitors your screen in real-time using `mss` and `OpenCV`, evaluates the content using a trained `MobileNetV2` model, and automatically minimizes target browser windows if prohibited content is detected.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run `stcb.py` to collect your own data (save screenshots to `dataset/safe` and `dataset/prohibited`).
4. Run the training script to generate your own `content_model.h5`.
5. Run `run_blocker.py` to start real-time monitoring.