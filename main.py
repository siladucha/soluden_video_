import cv2
import numpy as np
import multiprocessing
import logging
import time

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def streamer(video_path, frame_queue):
    """
    Captures a video stream, extracts frames, and sends them to the queue.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error opening video file")
        frame_queue.put(None)
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

    cap.release()
    frame_queue.put(None)  # Signal for termination
    logging.info("Video fully processed, terminating streamer.")

def detector(frame_queue, detection_queue, analytics_queue):
    """
    Performs motion detection using BackgroundSubtractor and collects analytics.
    """
    back_sub = cv2.createBackgroundSubtractorMOG2()
    movement_data = []
    start_time = time.time()
    frame_count = 0

    while True:
        frame = frame_queue.get()
        if frame is None:
            analytics_queue.put(movement_data)
            detection_queue.put(None)
            break

        fg_mask = back_sub.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 500]
        detection_queue.put((frame, detections))
        frame_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            movement_data.append((elapsed_time, len(detections)))
            logging.info(f"Motion density for {elapsed_time:.2f} sec: {len(detections)} objects")
            start_time = time.time()
            frame_count = 0

def apply_blur(frame, detections, blur_type='gaussian'):
    """
    Applies adaptive blurring to detected areas.
    blur_type: 'gaussian', 'median', 'bilateral'
    """
    for (x, y, w, h) in detections:
        roi = frame[y:y + h, x:x + w]

        if blur_type == 'gaussian':
            blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
        elif blur_type == 'median':
            blurred_roi = cv2.medianBlur(roi, 15)
        elif blur_type == 'bilateral':
            blurred_roi = cv2.bilateralFilter(roi, 15, 75, 75)
        else:
            blurred_roi = roi  # If blur type is not specified, do not apply

        frame[y:y + h, x:x + w] = blurred_roi
    return frame


def presenter(detection_queue, blur_type, draw_boxes=False):
    """
    Displays the video stream with blurring applied.

    Parameters:
    - detection_queue: Queue containing frames and detections.
    - blur_type: Type of blurring ('gaussian', 'median', 'bilateral').
    - draw_boxes: If True, draws bounding boxes around detected objects.
    """
    while True:
        data = detection_queue.get()
        if data is None:
            break

        frame, detections = data
        frame = apply_blur(frame, detections, blur_type)

        if draw_boxes:
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exit by pressing 'q'")
            break

    cv2.destroyAllWindows()
    logging.info("Presenter finished.")

def analytics_processor(analytics_queue):
    """
    Processes and outputs motion analytics.
    """
    movement_data = analytics_queue.get()
    total_time = sum(t for t, _ in movement_data)
    total_objects = sum(count for _, count in movement_data)
    avg_density = total_objects / total_time if total_time > 0 else 0
    logging.info(f"Average motion density: {avg_density:.2f} objects per second")

def main(video_path, blur_type='gaussian'):
    """
    Launches a multi-process video processing system with process control and analytics.
    """
    frame_queue = multiprocessing.Queue()
    detection_queue = multiprocessing.Queue()
    analytics_queue = multiprocessing.Queue()

    processes = [
        multiprocessing.Process(target=streamer, args=(video_path, frame_queue), name="Streamer"),
        multiprocessing.Process(target=detector, args=(frame_queue, detection_queue, analytics_queue), name="Detector"),
        multiprocessing.Process(target=presenter, args=(detection_queue, blur_type), name="Presenter"),
        multiprocessing.Process(target=analytics_processor, args=(analytics_queue,), name="Analytics")
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    logging.info("All processes have completed.")

if __name__ == "__main__":
    # Select blur type: 'gaussian', 'median', 'bilateral'

    video_path = "istockphoto-483534858-640_adpp_is.mp4"
    blur_type = 'gaussian'
    main(video_path, blur_type)

    video_path = "istockphoto-1140675444-640_adpp_is.mp4"
    blur_type = 'median'
    main(video_path, blur_type)