# Crime Detection Web App

A web application for detecting crime-related objects using YOLOv8 and MobileNet.

## Setup
1. Clone the repo: `git clone <your-repo-url>`
2. Activate virtual env: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `python app/app.py`

## Status
- **Day 1**: Project setup completed.
  - Initialized new Git repository.
  - Configured Codespaces with Python 3.9.
  - Set up virtual environment with core dependencies.
  - Created Flask app structure and minimal app.

  - Day 1 completed with Git, Codespaces, venv, and Flask setup.

- **Day 2**: Dataset collection and preprocessing completed.
  - Created data structure with 180 images across 6 classes.
  - Labeled images with YOLOv8 and remapped to custom classes.
  - Split into train (70%), val (15%), test (15%) sets.
  - Preprocessed images with augmentation pipeline.

  - Day 2 completed with dataset fully labeled, split, and preprocessed.
