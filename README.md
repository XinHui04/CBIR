# CBIR - Content-Based Image Retrieval

A Flask-based content-based image retrieval (CBIR) project for a nursing-home object dataset. The system extracts image features from a query photo and returns the most similar images from the dataset using either cosine similarity or Euclidean distance.

## Features

- Flask web interface for uploading a query image
- Search by visual similarity using three feature types:
  - Color
  - Texture
  - Shape
- Supports two similarity metrics:
  - Cosine similarity
  - Euclidean distance
- Category filtering in the UI
- Automatic feature database generation from the dataset
- Excludes unwanted classes during preprocessing and search, including:
  - human_being
  - fire_extinguisher

## Project Structure

```text
CBIR/
|-- app.py
|-- preprocess.py
|-- feature_database.py
|-- search.py
|-- color_feature.py
|-- texture_feature.py
|-- shape_feature.py
|-- requirements.txt
|-- feature_database.npz
|-- Dataset/
|-- templates/
|   `-- index.html
|-- static/
|   `-- uploads/
`-- uploads/
```

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Flask

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone or open the project folder.
2. Create and activate a virtual environment.
3. Install the dependencies.
4. Build the feature database.
5. Run the Flask app.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build the Feature Database

The file `feature_database.npz` is generated from the dataset. Rebuild it whenever the dataset changes.

```bash
python feature_database.py
```

This scans the `Dataset/` folder, extracts features for valid image files, and stores the results in `feature_database.npz`.

## Run the Web App

Start the Flask application with:

```bash
python app.py
```

Then open the local server URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## How It Works

1. A query image is uploaded through the web UI.
2. The app extracts color, texture, and shape features from the query image.
3. The same feature representation is loaded from the database for all dataset images.
4. The chosen similarity metric ranks the database images.
5. The top matching images are displayed with their category labels and scores.

## Supported Image Types

- JPG
- JPEG
- PNG
- BMP
- TIFF
- WEBP

## Notes

- The dataset currently contains multiple object categories under `Dataset/MYNursingHome/`.
- The preprocessing pipeline excludes non-target classes such as `human_being` and `fire_extinguisher`.
- If you change the dataset or feature extraction code, rebuild `feature_database.npz` before searching again.

## Example Use Case

Upload a room or furniture image, choose the desired metric and feature types, and the app will return visually similar items from the nursing-home dataset.

## License

No license file is included in this repository. Add one if you plan to publish or share the project.
