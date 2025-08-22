import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

def encode_faces(image_folder='faces'):
    known_encodings = []
    known_names = []

    # Traverse each person's folder inside 'known_faces'
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)
        
        # Check if it's a directory
        if os.path.isdir(person_folder):
            # Traverse each image in the person's folder
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)

                # Load the image using face_recognition
                image = face_recognition.load_image_file(img_path)

                # Convert image to RGB to avoid unsupported format errors
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Get encodings
                encodings = face_recognition.face_encodings(image)

                # If at least one encoding is found, add it to known_encodings
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                else:
                    print(f"Warning: No face detected in {img_path}")

    # Save the known encodings and names to .npy files for later use
    np.save('known_encodings.npy', known_encodings)
    np.save('known_names.npy', known_names)
    print("Encodings saved successfully.")

# Run the encoding function
encode_faces()

def recognize_faces_and_mark_attendance():
    # Load saved encodings and names
    try:
        known_encodings = np.load('encodings.npy', allow_pickle=True)
        known_names = np.load('names.npy', allow_pickle=True)
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return

    # Initialize a DataFrame to store attendance
    attendance = pd.DataFrame(columns=['Name', 'Time'])

    # Open the webcam for capturing frames
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame from webcam.")
            break

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Compare each detected face with the known encodings
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            # Compute face distances
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)

                # If a match is found, update the name
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            # Mark attendance only if the person is not already marked
            if name != "Unknown" and name not in attendance['Name'].values:
                new_row = pd.DataFrame({'Name': [name], 'Time': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]})
                attendance = pd.concat([attendance, new_row], ignore_index=True)

            # Draw a rectangle around the face and label it
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the video feed with recognized faces
        cv2.imshow('Face Recognition Attendance System', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Save the attendance record to a CSV file
    attendance.to_csv('attendance.csv', index=False)
    print("Attendance marked successfully.")

# Run face recognition
# recognize_faces_and_mark_attendance()