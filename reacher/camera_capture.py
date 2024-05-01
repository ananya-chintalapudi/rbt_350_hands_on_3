import numpy as np
import cv2

# Define the resolution and frame rate of the video recording
resolution_width = 1920  # Width of the video resolution
resolution_height = 1080  # Height of the video resolution
fps = 30  # Frame rate of the video recording

# Calculate the focal length based on the resolution and field of view
# Assuming a horizontal field of view of 60 degrees (common for smartphone cameras)
horizontal_fov_degrees = 60
focal_length = resolution_width / (2 * np.tan(np.deg2rad(horizontal_fov_degrees / 2)))

# Calculate the principal point (assuming it's at the center of the image)
principal_point = (resolution_width / 2, resolution_height / 2)

# Assume no distortion
distortion_coefficients = np.zeros((4, 1))

# Calculate the approximate intrinsic matrix
intrinsic_matrix = np.array([[focal_length, 0, principal_point[0]],
                             [0, focal_length, principal_point[1]],
                             [0, 0, 1]])

# Load your calibration parameters
# You should have saved these from a previous calibration step
# intrinsic_matrix, distortion_coefficients = ...

def transform_to_global_frame(point_2d):
    # Convert pixel coordinates to camera coordinates
    point_3d_camera_frame = cv2.convertPointsToHomogeneous(point_2d)
    point_3d_global = np.dot(np.linalg.inv(intrinsic_matrix), point_3d_camera_frame.reshape((3, 1)))
    
    return point_3d_global[:3]  # Return only x, y, z coordinates



cap = cv2.VideoCapture(0)

# Set the resolution and frame rate of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
cap.set(cv2.CAP_PROP_FPS, fps)

while(True):
    # Capture frame-by-frame
    ret, captured_frame = cap.read()
    output_frame = captured_frame.copy()

    # Convert original image to BGR, since Lab is only available from BGR
    captured_frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2BGR)
    # First blur to reduce noise prior to color space conversion
    captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 3)
    # Convert to Lab color space, we only need to check one channel (a-channel) for red here
    captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)
    # Threshold the Lab image, keep only the red pixels
    # Possible yellow threshold: [20, 110, 170][255, 140, 215]
    # Possible blue threshold: [20, 115, 70][255, 145, 120]
    captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([20, 150, 150]), np.array([190, 255, 255]))
    # Second blur to reduce more noise, easier circle detection
    captured_frame_lab_red = cv2.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)
    # Use the Hough transform to detect circles in the image
    circles = cv2.HoughCircles(captured_frame_lab_red, cv2.HOUGH_GRADIENT, 1, captured_frame_lab_red.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=60)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        center_x, center_y = circles[0, 0], circles[0, 1]
        
        # Transform the center point to global frame
        center_point_camera_frame = np.array([[center_x, center_y]], dtype=np.float32)
        center_point_global_frame = transform_to_global_frame(center_point_camera_frame)
        
        # Print or use the coordinates in the global frame
        print("Circle center in global frame:", center_point_global_frame)

        # Draw circle on output frame
        cv2.circle(output_frame, center=(center_x, center_y), radius=circles[0, 2], color=(0, 255, 0), thickness=2)

    # Display the resulting frame, quit with q
    cv2.imshow('frame', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()