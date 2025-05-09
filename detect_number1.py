# import cv2
# import numpy as np
# import glob
# # import easyocr
# #size of rectangle
# #(134,82) - (333,93) 
# #(128,116) - (329,124)
# def find_darkest_region(image_path,c):
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error loading image.")
#         return
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # black_mask = cv2.inRange(gray, 0, 110)
#     black_mask = cv2.adaptiveThreshold(gray, 255, 
#                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
#                                         cv2.THRESH_BINARY_INV, 
#                                         blockSize=51,C=10)
#     # copy of the original image (no coloring)
#     highlighted = image.copy()
#     highlighted[black_mask == 255] = [255, 255, 255] 
#     win_w, win_h = 200, 40
#     max_black_count = 0
#     best_coords = (0, 0)

#     for y in range(0, gray.shape[0] - win_h + 1, 5):
#         for x in range(0, gray.shape[1] - win_w + 1, 5):
#             window = black_mask[y:y + win_h, x:x + win_w]
#             black_count = cv2.countNonZero(window)

#             if black_count > max_black_count:
#                 max_black_count = black_count
#                 best_coords = (x, y)

#     x, y = best_coords
#     cv2.rectangle(highlighted, (x, y), (x + win_w, y + win_h), (0, 0, 255), 2)
#     new_image = highlighted[y:y + win_h, x:x + win_w]

#     # green_image = image.copy()
#     # green_image[black_mask == 255] = [0, 255, 255] 
#     # reader = easyocr.Reader(['en'])  # Loads English OCR model
#     # result = reader.readtext(new_image)
#     # numbers = ''.join([res[1] for res in result if res[1].isdigit()])
#     # print("Extracted Number:", numbers)


#     cv2.imwrite(f"new_image{c}.jpg", new_image)

#     while True:
#         cv2.imshow("Darkest Region Detected", new_image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cv2.destroyAllWindows()

# image_paths = glob.glob("*.jpg")
# c=100
# for path in image_paths:
#     print(path)
#     find_darkest_region(path,c)
#     c+=1
#     if c>=108:
#         break



import cv2
import numpy as np
import glob
from PIL import Image
import pytesseract
from matplotlib import pyplot as plt
def find_number_frame(image_path,c):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.adaptiveThreshold(gray, 255, 
                                       cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY_INV, 
                                        blockSize=51,C=10)

    # highlighted = image.copy()
    # highlighted[black_mask == 255] = [255, 255, 255] 
    win_w, win_h = 200, 40
    max_black_count = 0
    best_coords = (0, 0)

    for y in range(0, gray.shape[0] - win_h + 1, 5):
        for x in range(0, gray.shape[1] - win_w + 1, 5):
            window = black_mask[y:y + win_h, x:x + win_w]
            black_count = cv2.countNonZero(window)

            if black_count > max_black_count:
                max_black_count = black_count
                best_coords = (x, y)

    x, y = best_coords
    # cv2.rectangle(highlighted, (x, y), (x + win_w, y + win_h), (0, 0, 255), 2)
    # new_image = highlighted[y:y + win_h, x:x + win_w]
    #new method 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img[y:y + win_h, x:x + win_w]
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3) 
    edges = cv2.magnitude(sobel_x, sobel_y)
    edges = cv2.convertScaleAbs(edges)

    #number extraction
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary
    #type1
    binary_pil = Image.fromarray(binary)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(binary_pil, config=config)
    print("kamal_here is the extracted number: ", text.split())

    #type2
    # digit_width = binary.shape[1] // 5
    # digit_imgs = [binary[:, i*digit_width:(i+1)*digit_width] for i in range(5)]
    # digit_imgs_resized = [cv2.resize(d, (28, 28), interpolation=cv2.INTER_AREA) for d in digit_imgs]
    # templates = {}
    # for digit in range(10):
    #     template_path = f"number2/{digit}.jpg"  # update path if needed

    #     tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    #     if tmpl is not None:
    #         templates[digit] = tmpl
    # def match_digit(img, templates):
    #     max_score = -1
    #     best_digit = -1
    #     for digit, tmpl in templates.items():
    #         result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    #         score = result[0][0]
    #         if score > max_score:
    #             max_score = score
    #             best_digit = digit
    #         return best_digit
    # recognized_digits = [match_digit(digit, templates) for digit in digit_imgs_resized]
    # print("Recognized Digits:", recognized_digits)
    # plt.figure(figsize=(10, 2))
    # for i, d_img in enumerate(digit_imgs_resized):
    #     plt.subplot(1, 5, i+1)
    #     plt.imshow(d_img, cmap='gray')
    #     plt.title(f"Digit {i+1}")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    #for saving image
    # cv2.imwrite(f"number_frame/image{c}.jpg",edges)
    while True:
        cv2.imshow("Darkest Region Detected", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

image_paths = glob.glob("query_images/*.jpg")
c=0
for path in image_paths:
    print(path)
    find_number_frame(path,c)
    c+=1
