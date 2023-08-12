import cv2
import mediapipe as mp
import joblib
import tkinter as tk
from tkinter import scrolledtext
from collections import deque

def crop_hand_from_image(image, hand_landmarks):
    expansion_factor = 0.2  

    min_x = min([landmark.x for landmark in hand_landmarks]) * image.shape[1]
    max_x = max([landmark.x for landmark in hand_landmarks]) * image.shape[1]
    min_y = min([landmark.y for landmark in hand_landmarks]) * image.shape[0]
    max_y = max([landmark.y for landmark in hand_landmarks]) * image.shape[0]

    width = max_x - min_x
    height = max_y - min_y

    min_x = max(0, min_x - expansion_factor * width)
    max_x = min(image.shape[1], max_x + expansion_factor * width)
    min_y = max(0, min_y - expansion_factor * height)
    max_y = min(image.shape[0], max_y + expansion_factor * height)

    cropped_image = image[int(min_y):int(max_y), int(min_x):int(max_x)]
   
    return cropped_image

def extract_hand_landmarks(image, mp_hands):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            cropped_image = crop_hand_from_image(image, hand_landmarks.landmark)
            cv2.imshow('IMAGE',cropped_image)
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_results = mp_hands.process(cropped_rgb)

            if cropped_results.multi_hand_landmarks:
                cropped_hand_landmarks = cropped_results.multi_hand_landmarks[0]
            else:
                cropped_hand_landmarks = hand_landmarks  

            palm_landmark = (cropped_hand_landmarks.landmark[0].x, cropped_hand_landmarks.landmark[0].y)
            normalized_landmarks = [(landmark.x - palm_landmark[0], landmark.y - palm_landmark[1])
                                    for landmark in cropped_hand_landmarks.landmark]
            return normalized_landmarks
        else:
            return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def test_model_on_landmarks(hand_landmarks, model_path, label_encoder):
    flat_landmarks = [coord for point in hand_landmarks for coord in point]
    input_data = [flat_landmarks]
    predicted_value = model.predict(input_data)[0]

    confidences = []
    for i in range(len(label_encoder.classes_)):
        lower_boundary = label_encoder.transform([label_encoder.classes_[i]])[0] - 0.3
        upper_boundary = label_encoder.transform([label_encoder.classes_[i]])[0] + 0.3
        confidence = 1 - min(abs(predicted_value - lower_boundary), abs(predicted_value - upper_boundary))
        confidences.append(confidence)

    max_prob_idx = confidences.index(max(confidences))
    predicted_label = label_encoder.inverse_transform([max_prob_idx])[0]

    return predicted_label


def process_video_frame():
    ret, frame = video_capture.read()
    if ret:
        hand_landmarks = extract_hand_landmarks(frame, mp_hands)
        if hand_landmarks is not None:
            predicted_label = test_model_on_landmarks(hand_landmarks, model, label_encoder)
            prediction_buffer.append(predicted_label)

            if len(prediction_buffer) == buffer_size:
                most_frequent_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                message_text.config(state=tk.NORMAL)
                message_text.delete("1.0", tk.END)
                message_text.insert(tk.END, most_frequent_prediction + ' ')
                message_text.config(state=tk.DISABLED)

        
        
        if hand_landmarks is not None:
            for landmark in hand_landmarks:
                x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        
        combined_frame = cv2.hconcat([frame, frame])
        cv2.imshow('Hand Gesture Recognition - Comparison', combined_frame)

        cv2.waitKey(1)

        


root = tk.Tk()
root.title("Hand Gesture Recognition")

model_path = "trained_model.joblib"  
label_encoder = joblib.load("label_encoder.joblib")  
model = joblib.load(model_path)
video_capture = cv2.VideoCapture(0)  
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)

buffer_size = 10
prediction_buffer = deque(maxlen=buffer_size)
message_text = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD, state=tk.DISABLED)
message_text.pack(padx=10, pady=10)

process_video_frame()
root.mainloop()
video_capture.release()
cv2.destroyAllWindows()