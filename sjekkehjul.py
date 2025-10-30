import cv2
import numpy as np

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
    def __init__(self, real_wheelbase_cm=262):
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
    
    midx = (fx + bx) // 2
    midy = (fy + by) // 2
    
    displacement = front_tracker.get_displacement()
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


def main():
    video_path = "Larry_space/data_video/40kmt.mp4"
    
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"🚗 Real wheelbase: 262 cm")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('wheelbase_speed.mp4', fourcc, fps, (width, height))
    
    front_tracker = None
    back_tracker = None
    kf_wheelbase = Kalman1D(q=5, r=10)
    speed_calc = SpeedCalculator(real_wheelbase_cm=262)
    
    frame_num = 0
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
            
            wheelbase_px = back_tracker.x - front_tracker.x
            smooth_wb = kf_wheelbase.update(wheelbase_px)
            
            if not calibrated and smooth_wb > 100:
                speed_calc.calibrate(smooth_wb)
                calibrated = True
                print(f"\n✅ Calibrated: {smooth_wb:.1f} px = 262 cm ({speed_calc.px_to_cm:.3f} cm/px)")
            
            vis = visualize(frame, front_tracker, back_tracker, smooth_wb, speed_calc, fps)
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
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()