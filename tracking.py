
import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import os

def output_video(frames, output_filename):

    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]  # dimentions of the FPS
    output_fps = 30  # FPS in the output video

        # Build an object VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el formato de video MP4
    out = cv2.VideoWriter(os.path.join(output_filename, 'output_video.mp4'), fourcc, output_fps, (frame_width, frame_height))

        # Writte the FPS in the outputvideo
    for frame in frames:
        out.write(frame)

        # Close the object VideoWriter
    out.release()

if __name__ == '__main__':
    cap = cv2.VideoCapture(r"C:\Users\Usuario\Desktop\tracking\people.mp4")

    model = YOLO("yolov8n.pt")

    tracker = Sort()
    frame_count = 0
    frames = []
    track_label_dict = {}  # Dictionary for mapping track_id to labels

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        results = model(frame, stream=True)

        frame_count += 1

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.20)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            for i, (xmin, ymin, xmax, ymax, track_id) in enumerate(tracks):
                # Check if the track_id already has an associated label.
                if track_id in track_label_dict:
                    label = track_label_dict[track_id]
                else:
                    # If it doesn't have a label, assign a new label and save it in the dictionary.
                    label = res.names[filtered_indices[i]]
                    track_label_dict[track_id] = label

                cv2.putText(img=frame, text=f"Id: {track_id}, Class: {label}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        # Add the frame with the labels to the list of frames.
        frames.append(frame)

    cap.release()

    # Call the function to write the output video.
    output_filename = r"C:\Users\Usuario\Desktop\tracking"
    output_video(frames, output_filename)
