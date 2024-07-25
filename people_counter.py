import cv2
from ultralytics import YOLO
import argparse

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

if __name__ == "__main__":

    default_webcam = True
    if default_webcam:
        RTSP_ADDRESS = 0 # for inbuilt webcam
    else:
        # Add the rtsp address for the office camera
        RTSP_ADDRESS = "rtsp://admin:AIFabric1234@192.168.2.83:554/cam/realmonitor?channel=1&subtype=0"

    # Open the video stream
    cap = cv2.VideoCapture(RTSP_ADDRESS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, save = True, stream = True)
            for result in results:
                # Visualize the results on the frame
                annotated_frame = result.plot()
                print(result.__len__())

                # Added text to the annotated_frame
                cv2.putText(annotated_frame, f"Detected {result.__len__()} person(s) in the room",(frame_width//2 -60, 30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

                # Display the annotated frame
                cv2.imshow("People Counter Inference", annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


