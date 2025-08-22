-----

## Face Recognition Attendance System using Python 👨‍💻

This repository contains the source code for a **Face Recognition Attendance System** built with Python. The system is designed to automate the process of marking attendance by using facial recognition technology. It identifies students or employees from a live video stream or a static image and records their attendance in a CSV file, complete with timestamps.

-----

### Key Features ✨

  * **Real-Time Face Detection & Recognition:** Identifies multiple faces simultaneously from a live webcam feed.
  * **Automated Attendance Logging:** Automatically marks attendance and records the person's name and arrival time into a `.csv` file.
  * **Simple & Efficient:** Uses a clean and straightforward approach, making it easy to understand and modify.
  * **New User Enrollment:** Easily add new individuals to the system by simply adding their image to the designated folder.

-----

### How It Works ⚙️

The system operates in two main phases:

1.  **Encoding Known Faces:** First, it processes a directory of images containing known individuals. It detects faces in these images and generates 128-d facial embeddings (a unique numerical representation of each face) using the `face_recognition` library.
2.  **Recognition & Marking:** It then captures frames from a live webcam feed. For each frame, it detects all visible faces, computes their embeddings, and compares them against the known embeddings. If a match is found, it logs the person's name and the current timestamp into an attendance sheet (e.g., `Attendance.csv`). To avoid duplicate entries, it only marks a person's attendance once per session.

-----

### Technologies Used 🛠️

  * **Python:** The core programming language.
  * **OpenCV (`cv2`):** Used for capturing video from the webcam and handling image processing tasks.
  * **face\_recognition:** A powerful and simple library for recognizing and manipulating faces. It's built on top of `dlib`.
  * **NumPy:** For handling the numerical operations required for the facial embeddings.
  * **Pandas (Optional):** Can be used for more advanced data manipulation of the attendance CSV file.

-----
### Contribution 🤝

Feel free to fork this repository, open issues, and submit pull requests. Any contributions to improve the system are welcome\!
