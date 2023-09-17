import os
import gradio as gr
import cv2
import random
from ultralytics import YOLO
from tracker import Tracker
from utils import ID2LABEL, compute_color_for_labels


colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
          for j in range(10)]

detection_threshold = 0.7
model = YOLO('model_data/yolov8m.pt')


def traffic_counting(video):

    OBJECT_IDS = {"person": [], 
                  "bicycle": [], 
                  "car": [], 
                  "motocycle": [], 
                  "bus": [], 
                  "truck": [], 
                  "other": []}

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    # Define the codec and create a VideoWriter object
    output_video = "out_video/out.mp4"
    cap_out = cv2.VideoWriter(output_video, 
                              cv2.VideoWriter_fourcc(*'MJPG'), 
                              cap.get(cv2.CAP_PROP_FPS), 
                              (frame.shape[1], frame.shape[0]))

    tracker = Tracker()

    while ret:
        results = model.predict(frame)

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, class_id, score])
                

            tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                class_id = track.class_id

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (compute_color_for_labels(class_id)), 3)

                label_name = ID2LABEL[class_id] if class_id in ID2LABEL.keys() else "other"
                if track_id not in OBJECT_IDS[label_name]:
                    OBJECT_IDS[label_name].append(track_id)

                cv2.putText(frame,f"{label_name}-{track_id}", 
                            (int(x1) + 5, int(y1) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA )
    
        cap_out.write(frame)
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    return output_video, OBJECT_IDS


input_video = gr.Video(type="file")
output_video = gr.Video(type="file", label="Processed Video")

output_data = gr.Textbox(interactive=False)

demo = gr.Interface(traffic_counting,
                    inputs=input_video,
                    outputs=[output_video, output_data],
                    examples=[os.path.join('video', x) for x in os.listdir('video')],
                    cache_axamples=True
                    )


if __name__ == "__main__":
    demo.launch(share= False)