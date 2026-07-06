import re
import subprocess

import cv2
import easyocr
import numpy as np

PLATE_WIDTH_TARGET = 600  # normalize all images to the same scale so contour thresholds stay consistent
DEBUG = True  # True while recording/taking screenshots, False if you just want the final result


def _select_image_with_button():
    # A small window with a clear button, shown before the file browser pops
    # up, so it's obvious what the program is about to do.
    import tkinter
    from tkinter import filedialog, ttk

    selected = {"path": ""}

    def on_click():
        path = filedialog.askopenfilename(
            title="Select a car photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
        )
        if path:
            selected["path"] = path
            root.destroy()

    root = tkinter.Tk()
    root.title("ALPR - License Plate Scanner")
    root.configure(bg="#1e1e2e")
    root.resizable(False, False)

    window_width, window_height = 420, 200
    x = (root.winfo_screenwidth() - window_width) // 2
    y = (root.winfo_screenheight() - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Scan.TButton",
        font=("Helvetica", 11, "bold"),
        padding=10,
        background="#4f8cff",
        foreground="white",
        borderwidth=0,
    )
    style.map("Scan.TButton", background=[("active", "#3d75e0")])

    tkinter.Label(
        root,
        text="Automated License Plate Recognition",
        font=("Helvetica", 14, "bold"),
        bg="#1e1e2e",
        fg="white",
    ).pack(pady=(32, 6))
    tkinter.Label(
        root,
        text="Select a car photo to detect its license plate",
        font=("Helvetica", 10),
        bg="#1e1e2e",
        fg="#a0a0b0",
    ).pack(pady=(0, 20))
    ttk.Button(
        root, text="Select Photo to Scan License Plate", command=on_click, style="Scan.TButton"
    ).pack()

    root.mainloop()
    return selected["path"]


def _select_image_with_zenity():
    result = subprocess.run(
        [
            "zenity",
            "--file-selection",
            "--title=Select a car photo",
            "--file-filter=Images | *.jpg *.jpeg *.png",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def select_image_path():
    # Try tkinter first, fall back to zenity, then to a plain terminal prompt
    # so the script always has a way to get an image path.
    for picker in (_select_image_with_button, _select_image_with_zenity):
        try:
            path = picker()
            if path:
                return path
        except Exception:
            # Whatever the reason a picker failed (missing library, no display,
            # dialog error), just fall through to the next option in the chain.
            continue

    return input("Enter the path to a car photo: ").strip()


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def resize_image(image, target_width=PLATE_WIDTH_TARGET):
    height, width = image.shape[:2]
    scale = target_width / width
    return cv2.resize(image, (target_width, int(height * scale)))


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral filter reduces noise while preserving edges; a plain Gaussian blur
    # would also soften the plate's edges and make Canny's job harder.
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 30, 200)
    return gray, blurred, edges


def _rank_plate_contours(contours, image_width, image_height, max_candidates):
    # A license plate is a wide rectangle, a modest fraction of the frame,
    # roughly centered horizontally, and not near the very top of the photo.
    # Rectangularity (contour area vs. its bounding box area) is only used as
    # a scoring bonus, not a hard filter: trim or reflections next to a real
    # plate often fuse with its edge, so the traced contour can be a thin,
    # sparse shape even when its bounding box lines up with the plate.
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        aspect_ratio = w / float(h)
        width_ratio = w / float(image_width)
        height_ratio = h / float(image_height)
        y_ratio = y / float(image_height)
        center_x = (x + w / 2) / float(image_width)

        if not (2.0 <= aspect_ratio <= 9.0):
            continue
        if not (0.07 <= width_ratio <= 0.60):
            continue
        if not (0.015 <= height_ratio <= 0.18):
            continue
        if not (0.25 <= y_ratio <= 0.95):
            continue
        if not (0.15 <= center_x <= 0.85):
            continue

        rectangularity = cv2.contourArea(contour) / float(w * h)
        aspect_score = 1 / (1 + abs(aspect_ratio - 4.5))
        center_score = 1 / (1 + abs(center_x - 0.5))
        candidates.append((aspect_score + center_score + rectangularity, contour))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    return [contour for _, contour in candidates[:max_candidates]]


def find_plate_candidates(edges, max_candidates=15):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:200]
    image_height, image_width = edges.shape[:2]
    return _rank_plate_contours(contours, image_width, image_height, max_candidates)


def find_plate_candidates_morph(gray, max_candidates=15):
    # A second, independent candidate source. Closing the edge map bridges
    # small gaps in the plate's border (e.g. where trim or reflections
    # interrupt it), which sometimes isolates a cleaner rectangle than the
    # plain Canny contours above do.
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, None, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:200]
    image_height, image_width = gray.shape[:2]
    return _rank_plate_contours(contours, image_width, image_height, max_candidates)


def crop_plate(image, contour, padding=5):
    x, y, w, h = cv2.boundingRect(contour)
    height, width = image.shape[:2]
    x1, y1 = max(x - padding, 0), max(y - padding, 0)
    x2, y2 = min(x + w + padding, width), min(y + h + padding, height)
    return image[y1:y2, x1:x2]


def clean_plate_text(text):
    allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    cleaned = "".join(char for char in text.upper() if char in allowed_chars)
    return " ".join(cleaned.split())


def read_plate_text(cropped_plate, reader):
    # A single fixed threshold doesn't suit every lighting condition, so a
    # few variants are tried; the first one that reads as a valid plate wins.
    gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    gray_plate = cv2.resize(gray_plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray_plate = cv2.bilateralFilter(gray_plate, 9, 75, 75)
    _, thresholded = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresholded)

    fallback_text = None
    for variant in (gray_plate, thresholded, inverted):
        results = reader.readtext(
            variant,
            detail=0,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        if not results:
            continue

        text = clean_plate_text(" ".join(results))
        if normalize_plate_text(text):
            return text
        if text and fallback_text is None:
            fallback_text = text

    return fallback_text


PLATE_PATTERN = re.compile(r"(\d{2})([A-Z]{1,3})(\d{2,4})")


def normalize_plate_text(text):
    # Matches the Turkish plate format (province code + letters + digits),
    # which filters out OCR noise from pavement, shadows or trim.
    compact = text.replace(" ", "").upper()
    if len(compact) >= 2:
        # OCR sometimes reads a leading 0 (province code) as the letter O.
        compact = compact[:2].replace("O", "0") + compact[2:]

    match = PLATE_PATTERN.fullmatch(compact)
    if not match:
        return None
    return f"{match.group(1)} {match.group(2)} {match.group(3)}"


def to_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def resize_for_display(image, size=(400, 250)):
    return cv2.resize(to_bgr(image), size)


def add_title(image, title):
    titled = image.copy()
    cv2.rectangle(titled, (0, 0), (titled.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(titled, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return titled


def blank_panel(text, size=(400, 250)):
    panel = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(panel, text, (20, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return panel


def show_pipeline(original, gray, blurred, edges, detected_image, cropped_plate):
    top_row = np.hstack([
        add_title(resize_for_display(original), "Original"),
        add_title(resize_for_display(gray), "Grayscale"),
        add_title(resize_for_display(blurred), "Blurred"),
    ])
    bottom_row = np.hstack([
        add_title(resize_for_display(edges), "Canny Edges"),
        add_title(resize_for_display(detected_image), "Detected Plate"),
        add_title(resize_for_display(cropped_plate), "Cropped Plate"),
    ])
    dashboard = np.vstack([top_row, bottom_row])

    cv2.imshow("ALPR Pipeline", dashboard)
    cv2.imwrite("alpr_pipeline_result.jpg", dashboard)


def main():
    image_path = select_image_path()
    image = resize_image(load_image(image_path))
    gray, blurred, edges = preprocess(image)

    plate_candidates = find_plate_candidates(edges) + find_plate_candidates_morph(gray)
    if not plate_candidates:
        print("Could not detect a plate contour.")
        return

    reader = easyocr.Reader(["en"], gpu=False)

    # Keep the first candidate whose OCR result matches a plate pattern.
    # If none match, don't fall back to a "best guess" - showing a random
    # ground/trim contour as if it were the plate would be misleading.
    best_contour, best_text = None, None
    for contour in plate_candidates:
        raw_text = read_plate_text(crop_plate(image, contour), reader)
        normalized_text = normalize_plate_text(raw_text) if raw_text else None
        if normalized_text:
            best_contour, best_text = contour, normalized_text
            break

    if best_text:
        debug_image = image.copy()
        x, y, w, h = cv2.boundingRect(best_contour)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped_plate = crop_plate(image, best_contour)
    else:
        debug_image = blank_panel("No plate confidently detected")
        cropped_plate = blank_panel("N/A")

    if DEBUG:
        show_pipeline(image, gray, blurred, edges, debug_image, cropped_plate)

    if best_text:
        print(f"Detected license plate: {best_text}")
    else:
        print("Could not confidently detect a plate in this image.")

    if DEBUG:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
