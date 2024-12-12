import cv2
import tkinter as tk
from tkinter import messagebox
import os
import numpy as np
from PIL import Image, ImageTk

# Load known faces
known_face_encodings = []
known_face_names = []

# Replace with the actual path to your known faces folder
known_faces_folder = 'known_faces'

# Check if the known faces folder exists
if not os.path.exists(known_faces_folder):
    print(f"Error: The folder {known_faces_folder} does not exist.")

# Load known faces from the folder
for filename in os.listdir(known_faces_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.imread(os.path.join(known_faces_folder, filename))
        if image is None:
            print(f"Error loading image: {filename}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            roi_color = image[y:y + h, x:x + w]
            roi_gray = gray[y:y + h, x:x + w]
            if roi_color is not None:
                known_face_encodings.append(roi_color)
            known_face_names.append(os.path.splitext(filename)[0])

# Check if the known face encodings list is empty
if not known_face_encodings:
    print("No known faces loaded.")
else:
    print(f"Loaded {len(known_face_encodings)} known faces from {known_faces_folder} folder.")

# Initialize the Tkinter window
root = tk.Tk()
root.title('Face Recognition Security System')

# Create a label for displaying the recognized face
face_label = tk.Label(root, text='Recognized Face: Unknown', bg='gray', font=('Helvetica', 16))
face_label.pack(side=tk.BOTTOM, fill=tk.X)

# Initialize the camera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Function to update the video feed
def update_frame():
    ret, frame = video_capture.read()
    if ret:
        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PhotoImage
        img = cv2.resize(rgb_frame, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_color = frame[y:y + h, x:x + w]
            if roi_color is not None:
                for i, known_face_encoding in enumerate(known_face_encodings):
                    if np.array_equal(roi_color, known_face_encoding):
                        name = known_face_names[i]
                        face_label.config(text=f'Recognized Face: {name}')
                        # Trigger action (popup message)
                        messagebox.showinfo('Access Granted', 'Door Opened')

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Update the frame
    root.after(10, update_frame)

# Create a label for the video feed
video_label = tk.Label(root)
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
