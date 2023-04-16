import cv2 
import numpy as np
from time import sleep

min_width = 80  
min_height = 80  
pixel_offset = 6 
counting_line_position = 500
frame_delay = 60
detected_objects = []
car_count = 0


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Open the video file
cap = cv2.VideoCapture('car_video.mp4')

# Create a background subtractor object
background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If there are no more frames, exit the loop
    if not ret:
        break

    # Calculate the time to wait between frames
    time = float(1 / frame_delay)
    sleep(time)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_DEFAULT)

    # Apply background subtraction to detect moving objects
    subtracted = background_subtractor.apply(blur)

    # Dilate the resulting image to fill in holes and gaps
    dilated = cv2.dilate(subtracted, np.ones((5, 5)))

    # Apply a morphological closing operation to remove small objects and smooth the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    

    # Find the contours of the objects in the resulting image
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the detection line
    cv2.line(frame, (25, counting_line_position), (1200, counting_line_position), (255, 127, 0), 3)

    # Loop through each detected object
    for (i, c) in enumerate(contours):
        # Get the bounding rectangle of the object
        (x, y, w, h) = cv2.boundingRect(c)

        # Check if the bounding rectangle meets the minimum size requirements
        if w < min_width or h < min_height:
            continue

        # Draw a rectangle around the object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the centroid of the object
        centroid = get_centroid(x, y, w, h)

        # Add the centroid to the list of detections
        detected_objects.append(centroid)

        # Draw a circle at the centroid of the object
        cv2.circle(frame, centroid, 4, (0, 0, 255), -1)

        # Check if the object crosses the detection line
        for (x, y) in detected_objects:
            if y < (counting_line_position + pixel_offset) and y > (counting_line_position - pixel_offset):
                car_count += 1
                cv2.line(frame, (25, counting_line_position), (1200, counting_line_position), (0, 127, 255), 3)
                detected_objects.remove((x, y))
                print("Car is detected: " + str(car_count))

    cv2.putText(frame, "VEHICLE COUNT : " + str(car_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
    cv2.imshow("Detectar", closed)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
