# def build_and_train_model():
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
#     x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
#     y_train = to_categorical(y_train, 10)
#     y_test = to_categorical(y_test, 10)

#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(100, activation='relu'),
#         Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))
#     return model

# model = build_and_train_model()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_and_show_digits(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return

    # Step 1: Preprocess - Blur and Sobel
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    edges = cv2.convertScaleAbs(edges)

    # Step 2: Thresholding to binary
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary  # Invert if needed

    # Step 3: Segment image into 5 equal digit regions
    digit_width = binary.shape[1] // 5
    digit_imgs = [binary[:, i*digit_width:(i+1)*digit_width] for i in range(5)]

    # Step 4: Display all digits
    plt.figure(figsize=(10, 2))
    for i, d_img in enumerate(digit_imgs):
        plt.subplot(1, 5, i + 1)
        plt.imshow(d_img, cmap='gray')
        plt.title(f"Digit {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
image_path = "b&g/image7.jpg"  # Replace with your image path
process_and_show_digits(image_path)
