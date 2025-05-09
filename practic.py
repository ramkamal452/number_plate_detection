import cv2
import numpy as np

def find_and_highlight_high_intensity_patch(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Highlight all pixels in (100–255) range in green
    mask = cv2.inRange(gray, 130, 255)
    highlighted = image.copy()
    highlighted[mask == 255] = [0, 255, 0]  # Green

    # Sliding window settings
    win_w, win_h = 200, 40
    max_count = 0
    best_coords = (0, 0)

    for y in range(0, gray.shape[0] - win_h + 1, 5):
        for x in range(0, gray.shape[1] - win_w + 1, 5):
            window = mask[y:y + win_h, x:x + win_w]
            count = cv2.countNonZero(window)

            if count > max_count:
                max_count = count
                best_coords = (x, y)

    # Draw red rectangle on best region
    x, y = best_coords
    cv2.rectangle(highlighted, (x, y), (x + win_w, y + win_h), (0, 0, 255), 2)

    # Show the result
    while True:
        cv2.imshow("Highlight + Best Region", highlighted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Usage
image_path = "22042025122032-2800010024 (1).jpg"
# find_and_highlight_high_intensity_patch(image_path)



import cv2
import numpy as np

def highlight_black_pixels_and_rectangle(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate black pixels (0–50)
    _, black_mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Highlight all black pixels in green
    highlighted = image.copy()
    highlighted[black_mask == 255] = [0, 255, 0]  # Green

    # Find contours of connected black regions
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Draw rectangle around the largest black region
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle

    # Display result
    while True:
        cv2.imshow("Highlighted Black Pixels and Rectangle", highlighted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Usage
image_path = "22042025122032-2800010024 (1).jpg"
# highlight_black_pixels_and_rectangle(image_path)
