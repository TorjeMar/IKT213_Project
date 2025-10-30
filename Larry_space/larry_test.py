import cv2 as cv
import numpy as np
import time


def split_video_to_frames(video_path: str, output_folder: str) -> any:
    """ Takes a video file and splits it into individual frames saved as images in the specified output folder"""
    cap = cv.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


video = "IKT213_Project/Larry_space/data_video/test.mp4"
output_folder = "IKT213_Project/Larry_space/data_frames"


split_video_to_frames(video, output_folder)


img = cv.imread("IKT213_Project/Larry_space/data_frames/frame_0050.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(blur, 50, 150)
lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv.imshow('Detected Lines', img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('IKT213_Project/Larry_space/results/frame_0050_lines.jpg', img)
print("Line detection completed and saved as 'frame_0050_lines.jpg'")
# This code reads an image, processes it to detect lines using the Hough Transform,
# and then draws the detected lines on the original image before displaying and saving it.
