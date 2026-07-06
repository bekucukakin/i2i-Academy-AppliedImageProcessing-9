# ALPR - Automated License Plate Recognition

This project is a Python-based Automated License Plate Recognition (ALPR) application built with classical image processing techniques and OCR.

The application detects a license plate from a car photo, crops the detected plate region, reads the characters using EasyOCR, and prints the recognized plate text to the terminal.

Unlike deep learning-based object detectors, this project focuses on a traditional image processing pipeline: grayscale conversion, noise reduction, edge detection, contour analysis, candidate filtering, OCR, and plate-format validation.

## Features

- Load a car photo through a simple file picker
- Resize the image to a consistent width for more stable processing
- Convert the image to grayscale
- Reduce noise while preserving edges using a bilateral filter
- Detect edges with Canny Edge Detection
- Locate possible license plate regions using contour analysis
- Use morphological closing as a second candidate detection method
- Crop candidate plate regions from the original image
- Read candidate regions with EasyOCR
- Validate OCR output using the Turkish license plate format
- Display the full processing pipeline in a single dashboard window
- Save the dashboard output as `alpr_pipeline_result.jpg`

## How It Works

The program follows a step-by-step image processing pipeline:

1. **Image Loading and Resizing**
   The selected image is loaded with OpenCV and resized to a fixed width. This helps contour and size-based filters behave more consistently across different image resolutions.

2. **Preprocessing**
   The image is converted to grayscale, then denoised with a bilateral filter. This reduces unnecessary noise while keeping important edges visible.

3. **Edge Detection**
   Canny Edge Detection is applied to reveal object boundaries in the image.

4. **Plate Candidate Detection**
   The program searches for license plate candidates using contour analysis. Candidate regions are filtered and scored based on:
   - Aspect ratio
   - Width and height ratio
   - Vertical position
   - Horizontal center position
   - Rectangularity score

   A second detection pass using morphological closing is also applied. This helps recover plate-like rectangular regions when the plate border is broken by reflections, trim, or lighting conditions.

5. **OCR Processing**
   Each candidate region is cropped and enlarged before being sent to EasyOCR. Multiple OCR input variants are tested, including:
   - Grayscale crop
   - Otsu-thresholded crop
   - Inverted threshold crop

6. **Plate Format Validation**
   OCR results are cleaned and checked against the Turkish license plate format:

   ```regex
   \d{2}[A-Z]{1,3}\d{2,4}
   ```



## Result Output

If a valid plate is found, the recognized license plate is printed to the terminal. If no candidate confidently matches a plate format, the program reports that it could not detect a plate instead of showing a wrong region as the result.

## Project Structure

```
i2i-Academy-AppliedImageProcessing-9/
├── plate_recognizer.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The project uses:
- OpenCV
- EasyOCR
- NumPy
- PyTorch backend required by EasyOCR

EasyOCR may download its model files the first time the program runs, so the first execution can take longer than usual.

### 3. Linux note

On some Linux systems, `tkinter` may need to be installed separately for the file picker window:

```bash
sudo apt install python3-tk
```

If `tkinter` is not available, the program tries to fall back to `zenity`. If no graphical picker is available, the image path can still be entered manually in the terminal.

## Running the Application

```bash
python plate_recognizer.py
```

After running the script, a small window opens. Click **"Select Photo to Scan License Plate"**, then choose a car image from your computer.

## Output

When `DEBUG = True`, the program displays a single dashboard window containing the main pipeline stages:

- Original image
- Grayscale image
- Blurred image
- Canny edge result
- Detected plate region
- Cropped plate region

Clicking any panel enlarges it to fill the window for a closer look; clicking again returns to the full grid. The same dashboard is also saved as `alpr_pipeline_result.jpg`.

The final OCR result is printed in the terminal.

```

```

## Known Limitations

This project uses classical image processing, so it works best with clear, well-lit images where the license plate is visible and not too small.

The detection may become less reliable when:
- The plate is too far from the camera
- The image is blurry
- The plate is strongly angled
- There is heavy reflection or shadow
- The plate is partly blocked
- The surrounding trim or grille visually merges with the plate border

These are expected limitations of contour-based localization. Production-level ALPR systems usually combine image processing with trained object detection models for better robustness.

## Assignment Note

This project was developed for the i2i Academy Applied Image Processing assignment. The goal is to demonstrate a classical image processing pipeline that prepares an image, locates a license plate region mathematically, applies OCR, and prints the recognized plate text.
