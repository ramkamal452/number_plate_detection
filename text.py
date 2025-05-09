
# # no accuracy for this model
# import easyocr

# reader = easyocr.Reader(['en'])  # Loads English OCR model
# result = reader.readtext('edges106.jpg')

# # Filter only digits
# numbers = ''.join([res[1] for res in result if res[1].isdigit()])
# print("Extracted Number:", numbers)

# import cv2
# import numpy as np

# # Load templates (grayscale)
# templates = {str(i): cv2.imread(f'numbers/image{i}.jpg', 0) for i in range(10)}

# def detect_digits(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Threshold to get white-ish areas
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

#     results = []

#     for digit, template in templates.items():
#         template = cv2.resize(template, (20, 30))  # Or any size smaller than your input image
#         while True:
#             cv2.imshow("name",thresh)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)

#         loc = np.where(res >= 0.7)
#         for pt in zip(*loc[::-1]):
#             results.append((pt[0], digit))  # Store x-coordinate and digit


#     # Sort by position (left to right)
#     results.sort()
#     print("results,:",results)
#     return ''.join(d for _, d in results)


# number = detect_digits("numbers\image2.jpg")
# print("number : ",number)



# import cv2
# import pytesseract
# import numpy as np
# pytesseract.pytesseract.tesseract_cmd = r"C:/Users/ramkalagarla/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

# # Path to image
# image_path = "edges100.jpg"

# # Load image
# img = cv2.imread(image_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Denoise and threshold
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY_INV, 11, 2)

# # Morphological operations to clean noise
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# cv2.imwrite("googs2.jpg",opening)
# # Find contours
# contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Filter and sort contours (assuming 5 digits only)
# digit_regions = []
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     if 20 < h < 100 and 10 < w < 100:  # adjust based on image size
#         digit_regions.append((x, y, w, h))

# # Sort left to right
# digit_regions = sorted(digit_regions, key=lambda b: b[0])

# # If more than 5 contours, take best 5 by size or spacing logic
# if len(digit_regions) > 5:
#     digit_regions = digit_regions[:5]  # or use spacing-based filtering

# # Extract digits
# digits = ""
# for x, y, w, h in digit_regions:
#     roi = opening[y:y+h, x:x+w]
#     roi = cv2.resize(roi, (28, 28))  # normalize size if needed
#     # Use Tesseract on each digit
#     digit = pytesseract.image_to_string(roi, config='--psm 10 digits')
#     digits += digit.strip()
# if digits:
#     print("Extracted Number:", digits)
# else:
#     print("No digits found")

import easyocr
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Initialize EasyOCR reader (limit to numeric characters for better accuracy)
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if available
image_path = 'b&g/image7.jpg'

# Read the image
image = cv2.imread(image_path)

# OCR: Limit character set to digits only
results = reader.readtext(image_path, detail=1, paragraph=False, allowlist='0123456789')

# Draw results and print
for (bbox, text, conf) in results:
    print(f"Detected Number: {text}, Confidence: {conf:.2f}")
    pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(image, text, tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Display the image with bounding boxes
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Numbers")
plt.axis("off")
plt.show()
