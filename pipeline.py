import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import requests

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

class Kalman1D:
    def __init__(self, q=1e-2, r=1e-1):
        self.x = None
        self.P = 1.0
        self.Q = q
        self.R = r
    
    def update(self, z):
        if self.x is None:
            self.x = z
            return self.x
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x


class WheelTracker:
    def __init__(self, x, y, r):
        self.kf_x = Kalman1D(q=5, r=10)
        self.kf_y = Kalman1D(q=5, r=10)
        self.kf_r = Kalman1D(q=1, r=5)
        
        self.x = self.kf_x.update(x)
        self.y = self.kf_y.update(y)
        self.r = self.kf_r.update(r)
        
        self.lost_frames = 0
        self.max_lost = 10
        
        self.prev_x = x
        self.prev_y = y
    
    def update(self, x, y, r):
        self.prev_x = self.x
        self.prev_y = self.y
        
        self.x = self.kf_x.update(x)
        self.y = self.kf_y.update(y)
        self.r = self.kf_r.update(r)
        self.lost_frames = 0
        return True
    
    def miss(self):
        self.lost_frames += 1
    
    def is_valid(self):
        return self.lost_frames < self.max_lost
    
    def distance_to(self, x, y):
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def get_displacement(self):
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        return np.sqrt(dx**2 + dy**2)


def detect_wheels_hough(image, y_region_start=0.6):
    h, w = image.shape[:2]
    y_start = int(h * y_region_start)
    roi = image[y_start:h, :]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=25,
        maxRadius=100
    )
    
    wheels = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            wheels.append((int(x), int(y + y_start), int(r)))
    
    return wheels


def detect_wheels_contours(image, y_region_start=0.6):
    h, w = image.shape[:2]
    y_start = int(h * y_region_start)
    roi = image[y_start:h, :]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    
    edges = cv2.Canny(gray, 40, 120)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wheels = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 30000:
            continue
        
        if len(cnt) < 5:
            continue
        
        try:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            
            aspect_ratio = max(MA, ma) / (min(MA, ma) + 1e-6)
            if aspect_ratio < 2.0:
                r = (MA + ma) / 4
                if 25 < r < 100:
                    wheels.append((int(x), int(y + y_start), int(r)))
        except:
            continue
    
    return wheels


def detect_wheels_adaptive(image, y_region_start=0.6):
    h, w = image.shape[:2]
    y_start = int(h * y_region_start)
    roi = image[y_start:h, :]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wheels = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800 or area > 25000:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.4:
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if 25 < r < 100:
                wheels.append((int(x), int(y + y_start), int(r)))
    
    return wheels


def detect_wheels_combined(image):
    wheels_hough = detect_wheels_hough(image)
    wheels_contours = detect_wheels_contours(image)
    wheels_adaptive = detect_wheels_adaptive(image)
    
    all_wheels = wheels_hough + wheels_contours + wheels_adaptive
    
    if not all_wheels:
        return []
    
    clustered = []
    used = set()
    
    for i, w1 in enumerate(all_wheels):
        if i in used:
            continue
        
        cluster = [w1]
        for j, w2 in enumerate(all_wheels):
            if j <= i or j in used:
                continue
            
            dist = np.sqrt((w1[0] - w2[0])**2 + (w1[1] - w2[1])**2)
            if dist < 50:
                cluster.append(w2)
                used.add(j)
        
        avg_x = int(np.mean([w[0] for w in cluster]))
        avg_y = int(np.mean([w[1] for w in cluster]))
        avg_r = int(np.mean([w[2] for w in cluster]))
        
        clustered.append((avg_x, avg_y, avg_r))
    
    return clustered


def update_trackers(front_tracker, back_tracker, detections):
    if not detections:
        if front_tracker:
            front_tracker.miss()
        if back_tracker:
            back_tracker.miss()
        return front_tracker, back_tracker
    
    if front_tracker is None or back_tracker is None or \
       not front_tracker.is_valid() or not back_tracker.is_valid():
        if len(detections) >= 2:
            sorted_dets = sorted(detections, key=lambda d: d[0])
            front_tracker = WheelTracker(*sorted_dets[0])
            back_tracker = WheelTracker(*sorted_dets[-1])
            return front_tracker, back_tracker
        else:
            return front_tracker, back_tracker
    
    front_match = None
    back_match = None
    front_dist = float('inf')
    back_dist = float('inf')
    
    for det in detections:
        x, y, r = det
        
        dist_front = front_tracker.distance_to(x, y)
        if dist_front < 150 and dist_front < front_dist:
            front_dist = dist_front
            front_match = det
        
        dist_back = back_tracker.distance_to(x, y)
        if dist_back < 150 and dist_back < back_dist:
            back_dist = dist_back
            back_match = det
    
    if front_match and back_match and front_match == back_match:
        if front_dist < back_dist:
            back_match = None
        else:
            front_match = None
    
    if front_match:
        front_tracker.update(*front_match)
    else:
        front_tracker.miss()
    
    if back_match:
        back_tracker.update(*back_match)
    else:
        back_tracker.miss()
    
    return front_tracker, back_tracker


class SpeedCalculator:
    def __init__(self, real_wheelbase_cm):
        self.real_wheelbase_cm = real_wheelbase_cm
        self.px_to_cm = None
        self.speed_history = []
        self.max_history = 10
        self.kf_speed = Kalman1D(q=5, r=15)
    
    def calibrate(self, wheelbase_px):
        if wheelbase_px > 0:
            self.px_to_cm = self.real_wheelbase_cm / wheelbase_px
    
    def calculate_speed(self, displacement_px, fps):
        if self.px_to_cm is None or fps <= 0:
            return None
        
        displacement_cm = displacement_px * self.px_to_cm
        
        speed_cm_per_sec = displacement_cm * fps
        speed_m_per_sec = speed_cm_per_sec / 100
        speed_kmh = speed_m_per_sec * 3.6
        
        if speed_kmh < 0 or speed_kmh > 200:
            return None
        
        smooth_speed = self.kf_speed.update(speed_kmh)
        
        self.speed_history.append(smooth_speed)
        if len(self.speed_history) > self.max_history:
            self.speed_history.pop(0)
        
        return smooth_speed
    
    def get_average_speed(self):
        if not self.speed_history:
            return None
        return np.mean(self.speed_history)


def visualize(frame, front_tracker, back_tracker, wheelbase, speed_calc, fps):
    img = frame.copy()
    
    if not front_tracker or not back_tracker:
        return img
    
    if not front_tracker.is_valid() or not back_tracker.is_valid():
        return img
    
    fx, fy, fr = int(front_tracker.x), int(front_tracker.y), int(front_tracker.r)
    bx, by, br = int(back_tracker.x), int(back_tracker.y), int(back_tracker.r)
    
    front_color = (0, 255, 0) if front_tracker.lost_frames == 0 else (0, 180, 255)
    back_color = (0, 255, 0) if back_tracker.lost_frames == 0 else (0, 180, 255)
    
    cv2.circle(img, (fx, fy), fr, front_color, 3)
    cv2.circle(img, (bx, by), br, back_color, 3)
    
    cv2.circle(img, (fx, fy), 5, (255, 255, 255), -1)
    cv2.circle(img, (bx, by), 5, (255, 255, 255), -1)
    
    cv2.line(img, (fx, fy), (bx, by), (0, 0, 255), 4)
    
    mid_x = (front_tracker.x + back_tracker.x) / 2
    mid_y = (front_tracker.y + back_tracker.y) / 2

    mid_dx = mid_x - visualize.prev_mid_x if hasattr(visualize, "prev_mid_x") else 0
    mid_dy = mid_y - visualize.prev_mid_y if hasattr(visualize, "prev_mid_y") else 0
    visualize.prev_mid_x, visualize.prev_mid_y = mid_x, mid_y

    displacement = np.sqrt(mid_dx**2 + mid_dy**2)

    current_speed = speed_calc.calculate_speed(displacement, fps)

    avg_speed = speed_calc.get_average_speed()
    
    panel_height = 140
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (350, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    y_offset = 35
    cv2.putText(img, f"Wheelbase: {int(wheelbase)} px = 262 cm",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 30
    if speed_calc.px_to_cm:
        cv2.putText(img, f"Scale: {speed_calc.px_to_cm:.3f} cm/px",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    y_offset += 30
    if current_speed is not None:
        speed_color = (0, 255, 0) if current_speed < 60 else (0, 165, 255)
        cv2.putText(img, f"Speed: {current_speed:.1f} km/h",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
    
    y_offset += 30
    if avg_speed is not None:
        cv2.putText(img, f"Avg Speed: {avg_speed:.1f} km/h",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)
    
    if front_tracker.lost_frames > 0 or back_tracker.lost_frames > 0:
        status = f"Lost: F={front_tracker.lost_frames} B={back_tracker.lost_frames}"
        cv2.putText(img, status, (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    
    return img


def detect_plate_number(image_path):
    # Load and display original image
    image = cv2.imread(image_path)
    
    # Convert to grayscale and display
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection with adaptive parameters
    edges = cv2.Canny(blurred, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    plate_contour = None
    for i, contour in enumerate(contours[:120]):
        perim = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perim, True)
        x,y,w,h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)
        rect_area = max(1, w*h)
        extent = area / rect_area
        ar = w / float(h) if h>0 else 0
        print(i, len(approx), "area", int(area), "w", w, "h", h, "ar", round(ar,2), "extent", round(extent,2))
        if len(approx) == 4 and 2.0 <= ar <= 6.0 and 300 < area < 120000 and extent > 0.5:
            plate_contour = approx
            print("Selected index", i)
            break
    
    if plate_contour is not None:
        image_with_plate = image.copy()
        cv2.drawContours(image_with_plate, [plate_contour], -1, (0, 0, 255), 3)
        
        # Extract plate region with padding
        x, y, w, h = cv2.boundingRect(plate_contour)
        padding = 5
        y_start = max(0, y - padding)
        y_end = min(gray.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(gray.shape[1], x + w + padding)
        
        plate_image = gray[y_start:y_end, x_start:x_end]
        
        # Enhanced preprocessing for OCR
        plate_image = cv2.resize(plate_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        plate_image = cv2.GaussianBlur(plate_image, (3, 3), 0)
        _, thresh = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # OCR with multiple configurations
        configs = [
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
        
        results = []
        for config in configs:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            if text:
                results.append(text)
        
        # Return the longest result or first non-empty
        plate_number = max(results, key=len) if results else "No text detected"
        print(f"OCR Results: {results}")
        return plate_number
    else:
        return None
    
def plate_finder(cap, frame_interval, frame_number, fps):
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process only every Nth frame
        if frame_number % frame_interval == 0:
            # Save frame temporarily
            temp_image_path = 'temp_frame.jpg'
            cv2.imwrite(temp_image_path, frame)
            
            print(f"\nProcessing frame {frame_number} (time: {frame_number/fps:.2f}s)")
            plate_number = detect_plate_number(temp_image_path)
            # make sure that the first character should be a letter and if its a number it should be removed and printed without
            if plate_number == None:
                print("No plate detected in this frame.")
            elif len(plate_number) == 8:
                plate_number = plate_number[1:]  # Remove first character
                print("Detected Plate Number:", plate_number)
                break
            elif plate_number and plate_number[0].isalpha():
                print(len(plate_number))
                print("Detected Plate Number:", plate_number)
                break
            else:
                print("Detected Plate Number:", plate_number.lstrip('0123456789'))
                break

        frame_number += 1
    return plate_number, frame_number

def speed_from_video(cap, length, frame, fps, width, height, total_frames):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = 1
    out = cv2.VideoWriter('wheelbase_speed.mp4', fourcc, output_fps, (width, height))

    front_tracker = None
    back_tracker = None
    kf_wheelbase = Kalman1D(q=5, r=10)
    speed_calc = SpeedCalculator(real_wheelbase_cm=length)

    recording = False
    stable_frames = 0
    unstable_frames = 0
    STABLE_REQUIRED = 10     # start saving after 10 stable frames
    UNSTABLE_LIMIT = 10      # stop saving after losing for 10 frames

    
    frame_num = frame
    calibrated = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        detections = detect_wheels_combined(frame)
        
        front_tracker, back_tracker = update_trackers(front_tracker, back_tracker, detections)
        
        if front_tracker and back_tracker and \
           front_tracker.is_valid() and back_tracker.is_valid():
            
            wheelbase_px = np.sqrt((back_tracker.x - front_tracker.x)**2 + (back_tracker.y - front_tracker.y)**2)
            smooth_wb = kf_wheelbase.update(wheelbase_px)
            speed_calc.px_to_cm = speed_calc.real_wheelbase_cm / smooth_wb
            
            if not calibrated and smooth_wb > 100:
                speed_calc.calibrate(smooth_wb)
                calibrated = True
                print(f"\n✅ Calibrated: {smooth_wb:.1f} px = {speed_calc.real_wheelbase_cm:.1f} cm ({speed_calc.px_to_cm:.3f} cm/px)")
            
            vis = visualize(frame, front_tracker, back_tracker, smooth_wb, speed_calc, fps)

            # Only save if both wheels have not been lost recently
            if front_tracker.lost_frames == 0 and back_tracker.lost_frames == 0:
                out.write(vis)

            cv2.imshow('Wheel Tracker - Speed Measurement', vis)

        else:
            debug_frame = frame.copy()
            for x, y, r in detections:
                cv2.circle(debug_frame, (x, y), r, (255, 0, 255), 2)
            cv2.putText(debug_frame, "Searching for wheels...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out.write(debug_frame)
            cv2.imshow('Wheel Tracker - Speed Measurement', debug_frame)
        
        if frame_num % 30 == 0:
            print(f"Frame {frame_num}/{total_frames} ({100*frame_num/total_frames:.1f}%)", end='\r')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    # Open the video file
    cap = cv2.VideoCapture('Larry_space/data_video/40kmt.mp4')

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps


    print(f"Video properties: {frame_count} frames, {fps} fps, {duration:.2f} seconds")

    # Process every Nth frame (adjust frame_interval to check more/fewer frames)
    frame_interval = 5  # Check every 30 frames (every 0.5 seconds at 60fps)
    frame_number = 0

    plate_number, frame = plate_finder(cap, frame_interval, frame_number, fps)

    regnr = plate_number
    r = requests.get('https://kjoretoyoppslag.atlas.vegvesen.no/ws/no/vegvesen/kjoretoy/kjoretoyoppslag/v2/oppslag/raw/'+ regnr)
    data = r.json()
    length = data['kjoretoy']['godkjenning']['tekniskGodkjenning']['tekniskeData']['akslinger']['akselGruppe'][0]['akselListe']['aksel'][0]['avstandTilNesteAksling'] / 10

    speed_from_video(cap, length, frame, fps, width, height, frame_count)

    cap.release()


if __name__ == "__main__":
    main()