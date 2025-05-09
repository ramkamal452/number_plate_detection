# import cv2
# import matplotlib.pyplot as plt

# # Load the image in grayscale
# img = cv2.imread('22042025123155-2800010031.jpg', cv2.IMREAD_GRAYSCALE)

# # Apply Gaussian Blur to reduce noise
# blurred = cv2.GaussianBlur(img, (5, 5), 0)

# # Apply Canny edge detection
# edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# # Show result (you can use cv2.imshow if not in Jupyter)
# plt.imshow(edges, cmap='gray')
# plt.title("Canny Edge Detection")
# plt.axis('off')
# plt.show()



# import cv2
# import matplotlib.pyplot as plt
# def fun(img):
#     # Load the image in grayscale
#     img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

#     # Apply Gaussian Blur to reduce noise
#     blurred = cv2.GaussianBlur(img, (5, 5), 0)

#     # Apply Sobel edge detection in the X and Y directions
#     sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
#     sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction

#     # Combine the gradients (magnitude of the edge)
#     edges = cv2.magnitude(sobel_x, sobel_y)

#     # Convert back to uint8 for proper display
#     edges = cv2.convertScaleAbs(edges)

#     # Show result (you can use cv2.imshow if not in Jupyter)
#     plt.imshow(edges, cmap='gray')
#     plt.title("Sobel Edge Detection")
#     plt.axis('off')
#     plt.show()
# # fun("22042025123155-2800010031.jpg")




import cv2
import numpy as np
import glob
from PIL import Image
import pytesseract
from matplotlib import pyplot as plt

def remove_shadows_and_adjust_exposure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_removed = cv2.divide(gray, dilated, scale=255)
    img_float = bg_removed.astype(np.float32) / 255.0
    img_float = np.power(img_float, 1.5)
    result = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    return result

def find_number_frame(image_path, c):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Couldn't load image at {image_path}")
        return

    print(f"[INFO] Processing: {image_path}")
    
    # Step 1: Shadow removal and exposure fix
    cleaned = remove_shadows_and_adjust_exposure(image)
    
    # Step 2: Threshold for dark region detection
    black_mask = cv2.adaptiveThreshold(
        cleaned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=51, C=10
    )

    # Step 3: Sliding window to find darkest region
    win_w, win_h = 200, 40
    max_black_count, best_coords = 0, (0, 0)

    for y in range(0, cleaned.shape[0] - win_h + 1, 5):
        for x in range(0, cleaned.shape[1] - win_w + 1, 5):
            window = black_mask[y:y + win_h, x:x + win_w]
            black_count = cv2.countNonZero(window)
            if black_count > max_black_count:
                max_black_count = black_count
                best_coords = (x, y)

    x, y = best_coords

    # Step 4: Crop and edge-detect the number region
    number_region = cleaned[y:y + win_h, x:x + win_w]
    blurred = cv2.GaussianBlur(number_region, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

    # Step 5: Binary + inversion
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary

    # Step 6: OCR using Tesseract
    binary_pil = Image.fromarray(binary)
    config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(binary_pil, config=config)
    digits = text.split()
    print(f"kamal_here is the extracted number: {digits}")

    # Step 7: Optional display
    plt.imshow(binary, cmap='gray')
    plt.title(f"Detected Region (image {c})")
    plt.axis('off')
    plt.show()

# Batch process all images in folder
image_paths = glob.glob("query_images/*.jpg")
for idx, path in enumerate(sorted(image_paths)):
    find_number_frame(path, c=idx)
