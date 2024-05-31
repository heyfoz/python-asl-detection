import cv2 # For image processing such as reading, resizing, and manipulation
import numpy as np # To use numerical operations
import tensorflow as tf # To use tensorflow operations
import string # To create list of letter classes (A-Z)
import mediapipe as mp # For hand tracking 

# Define colors
dark_blue = (48, 31, 27)  # Dark blue in BGR
white = (255, 255, 255)  # White in BGR

# Font and text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
text_position = (50, 50)

# Initialize MediaPipe Hands
print("Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
print("MediaPipe Hands initialized successfully.")

# Load the trained model
print("Loading the trained model...")
# Windows model path formatting example: 'C:\\Users\\username\\path\\to\\model.keras'
model_path = 'path/to/pretrained/model.keras'
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Initialize webcam
print("Initializing webcam...")
cap = cv2.VideoCapture(0)
print("Webcam initialized successfully.")

# Box parameters (UI size)
ui_box_size = 400
box_top_left = (50, 50)
box_bottom_right = (box_top_left[0] + ui_box_size, box_top_left[1] + ui_box_size)

# Processing size (model input size)
processing_size = 200
enlargement_factor = 4

# # Generate class names for A-Z, A_flipped through Z_flipped, and Blank
# class_names = list(string.ascii_uppercase) + [f"{char}_flipped" for char in string.ascii_uppercase] + ['Blank']

# Generate class names for A-Z
class_names = list(string.ascii_uppercase)

# Prediction buffer parameters
prediction_buffer = []
buffer_size = 10
consensus_threshold = 8

def preprocess_image(frame, hands):
    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    height, width, _ = frame.shape
    hand_mask = np.zeros((height, width), dtype=np.uint8)

    for hand_landmarks in results.multi_hand_landmarks:
        points = []
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            points.append((x, y))
        cv2.fillPoly(hand_mask, [np.array(points, dtype=np.int32)], 255)

    kernel = np.ones((5, 5), np.uint8)
    hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=1)
    hand_mask_blurred = cv2.GaussianBlur(hand_mask_dilated, (21, 21), 0)

    hand_on_black_background = np.where(hand_mask_blurred[..., None] > 0, frame, np.zeros_like(frame))

    # Resize and center the hand area in a new 200x200 image
    contours, _ = cv2.findContours(hand_mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x_min, x_max, y_min, y_max = width, 0, height, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)

    hand_area = hand_on_black_background[y_min:y_max, x_min:x_max]
    hand_width, hand_height = x_max - x_min, y_max - y_min
    scale_factor = min(processing_size / hand_width, processing_size / hand_height) * 0.8
    scaled_w, scaled_h = int(hand_width * scale_factor), int(hand_height * scale_factor)
    resized_hand_area = cv2.resize(hand_area, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    centered_image = np.zeros((processing_size, processing_size, 3), dtype=np.uint8)
    x_offset = (processing_size - scaled_w) // 2
    y_offset = (processing_size - scaled_h) // 2
    centered_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_hand_area

    # Convert to grayscale and expand to 3 channels
    grayscale_frame = cv2.cvtColor(centered_image, cv2.COLOR_BGR2GRAY)
    expanded_frame = np.stack((grayscale_frame,) * 3, axis=-1)

    return expanded_frame

try:
    print("Starting the main loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame from webcam. Exiting the loop.")
            break

        frame = cv2.flip(frame, 1)
        print("Frame flipped successfully.")  # Add this line

        processed_frame = preprocess_image(frame, hands)
        print("Frame processed successfully.")  # Add this line

        if processed_frame is not None:
            print("Predicting...")  # Add this line

            prediction_frame = np.expand_dims(np.float32(processed_frame) / 255.0, axis=0)
            prediction = model.predict(prediction_frame)
            predicted_index = np.argmax(prediction)

            prediction_buffer.append(predicted_index)
            if len(prediction_buffer) > buffer_size:
                prediction_buffer.pop(0)

            # Determine the text to display
            display_text = "Stabilizing..."
            if prediction_buffer.count(predicted_index) >= consensus_threshold:
                display_text = class_names[predicted_index]

            # Calculate text size and draw background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
            cv2.rectangle(frame, (text_position[0] - 5, text_position[1] - text_height - 5),
                        (text_position[0] + text_width + 5, text_position[1] + baseline + 5), white, -1)

            # Draw the text
            cv2.putText(frame, display_text, text_position, font, font_scale, dark_blue, thickness)

            enlarged_frame = cv2.resize(processed_frame, (processing_size * enlargement_factor, processing_size * enlargement_factor))
            cv2.imshow('Preprocessed Input', enlarged_frame)
        else:
            enlarged_empty_frame = np.zeros((processing_size * enlargement_factor, processing_size * enlargement_factor, 3), dtype=np.uint8)
            cv2.imshow('Preprocessed Input', enlarged_empty_frame)

        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key detected. Exiting the loop.")
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Resources released successfully.")    
