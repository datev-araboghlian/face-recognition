import cv2
import tkinter as tk
from tkinter import messagebox
import os
import numpy as np
from PIL import Image, ImageTk

# Load known faces and names
known_face_encodings = []
known_face_names = []

# Replace with the actual path to your known faces folder
known_faces_folder = 'known_faces'

# Create the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load known faces from the folder
for filename in os.listdir(known_faces_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.imread(os.path.join(known_faces_folder, filename))
        if image is None:
            print(f"Error loading image: {filename}")
            continue
        # Convert to grayscale for recognition
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        known_face_encodings.append(gray_image)
        known_face_names.append(os.path.splitext(filename)[0])

# Train the recognizer with the known faces
face_recognizer.train(known_face_encodings, np.array(range(len(known_face_names))))

# Initialize the Tkinter window
root = tk.Tk()
root.title('Face Recognition Security System')

# Center the window on the screen
window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_right = int(screen_width / 2 - window_width / 2)
position_down = int(screen_height / 2 - window_height / 2)
root.geometry(f'{window_width}x{window_height}+{position_right}+{position_down}')

# Create a full window frame for the color band
full_window_frame = tk.Frame(root, bg='red')
full_window_frame.pack(fill=tk.BOTH, expand=True)

# Create a label for displaying the recognized face
face_label = tk.Label(full_window_frame, text='Recognized Face: Unknown', bg='gray', font=('Helvetica', 16))
face_label.pack(side=tk.BOTTOM, fill=tk.X)

# Create a label for door status
status_label = tk.Label(full_window_frame, text='', bg='gray', font=('Helvetica', 16))
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# Initialize the camera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to update the video feed
def update_frame():
    ret, frame = video_capture.read()
    if ret:
        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get the original dimensions
        height, width, _ = rgb_frame.shape
        # Resize while maintaining aspect ratio
        aspect_ratio = width / height
        new_width = 640
        new_height = int(new_width / aspect_ratio)
        img = cv2.resize(rgb_frame, (new_width, new_height))
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Debugging: Print the number of faces detected
        print(f'Detected faces: {len(faces)}')

        # Draw rectangles around the faces on the original frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Prepare the region of interest for recognition
            roi_gray = gray[y:y + h, x:x + w]
            label, confidence = face_recognizer.predict(roi_gray)
            name = "Unknown"
            if confidence < 100:
                name = known_face_names[label]
                face_label.config(text=f'Hello {name} and Welcome')
                full_window_frame.config(bg='green')
                status_label.config(text='Door Opened')
            else:
                face_label.config(text='Recognized Face: Unknown')
                full_window_frame.config(bg='red')
                status_label.config(text='')

    # Update the frame
    root.after(10, update_frame)

# Create a label for the video feed
video_label = tk.Label(full_window_frame)
video_label.pack()

# Start updating frames
update_frame()

# Start the Tkinter main loop
root.mainloop()

# Release the capture
video_capture.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    print('Facial Recognition Security System')

#hello