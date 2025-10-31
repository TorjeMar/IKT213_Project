import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import requests

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Path to the Tesseract executable

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
        plt.tight_layout()
        plt.show()
        return None
    
def plate_finder(image):
    plate_number, frame = detect_plate_number(image)
    return plate_number, frame

def main():
    # Open the video file
    cap = cv2.VideoCapture('Larry_space/data_video/40kmt.mp4')

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video properties: {frame_count} frames, {fps} fps, {duration:.2f} seconds")

    # Process every Nth frame (adjust frame_interval to check more/fewer frames)
    frame_interval = 5  # Check every 30 frames (every 0.5 seconds at 60fps)
    frame_number = 0

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

    cap.release()
    print("\nVideo processing complete!")
    regnr = plate_number
    r = requests.get('https://kjoretoyoppslag.atlas.vegvesen.no/ws/no/vegvesen/kjoretoy/kjoretoyoppslag/v2/oppslag/raw/'+ regnr)
    data = r.json()
    bredth = data['kjoretoy']['godkjenning']['tekniskGodkjenning']['tekniskeData']['akslinger']['akselGruppe'][0]['akselListe']['aksel'][0]['avstandTilNesteAksling']
    print(f"Avstand mellom aksel : {bredth / 10} cm")

if __name__ == "__main__":
    main()