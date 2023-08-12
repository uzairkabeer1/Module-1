import cv2
import mediapipe as mp
import os
import csv


def extract_hand_landmarks(image_path, mp_hands):
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            palm_landmark = (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)
            
            normalized_landmarks = [(landmark.x - palm_landmark[0], landmark.y - palm_landmark[1])
                                    for landmark in hand_landmarks.landmark]
            return normalized_landmarks
        else:
            return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None



def process_dataset(dataset_folder, output_csv):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Label'] + [f'Keypoint_{i}' for i in range(1, 22)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for label in os.listdir(dataset_folder):
            label_folder = os.path.join(dataset_folder, label)
            print(label_folder)

            for image_file in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_file)
                print(image_path)
                hand_landmarks = extract_hand_landmarks(image_path, mp_hands)
                if hand_landmarks is not None:
                    row_data = {'Label': label}
                    for i, (x, y) in enumerate(hand_landmarks):
                        row_data[f'Keypoint_{i+1}'] = f'{x},{y}'
                    writer.writerow(row_data)

    
    mp_hands.close()

if __name__ == "__main__":
    train_data_folder = 'C:/Sign Language Detection/Module 1 (sign to text)/dataset/Train_Alphabet'
    test_data_folder = 'C:/Sign Language Detection/Module 1 (sign to text)/dataset/Test_Alphabet'
    train_output_csv = '../train_data.csv'
    test_output_csv = '../test_data.csv'

    process_dataset(test_data_folder, test_output_csv)
    process_dataset(train_data_folder, train_output_csv)
