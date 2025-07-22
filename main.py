import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Road background


def create_road_background(width, height):
    # Green background for grass
    road_img = np.zeros((height, width, 3), dtype=np.uint8)
    road_img[:] = (60, 180, 60)  # Grass green

    # Draw the road
    road_width = width // 2
    road_left = (width - road_width) // 2
    road_top = 0
    road_bottom = height

    # Road surface
    cv2.rectangle(road_img, (road_left, road_top),
                  (road_left + road_width, road_bottom), (80, 80, 80), -1)

    # Road markings
    marking_height = 30
    marking_width = 15
    marking_gap = 40

    for y in range(marking_height, height, marking_height + marking_gap):
        cv2.rectangle(road_img,
                      (width // 2 - marking_width // 2, y),
                      (width // 2 + marking_width // 2, y + marking_height),
                      (255, 255, 255), -1)

    # Road borders
    cv2.rectangle(road_img, (road_left, road_top),
                  (road_left + 10, road_bottom), (200, 150, 0), -1)
    cv2.rectangle(road_img, (road_left + road_width - 10, road_top),
                  (road_left + road_width, road_bottom), (200, 150, 0), -1)

    return road_img

# Car image


def create_car_image(width, height):
    car_img = np.zeros((height, width, 4), dtype=np.uint8)

    # Car body
    cv2.rectangle(car_img, (0, 0), (width, height), (0, 100, 255, 255), -1)

    # Car top
    cv2.rectangle(car_img, (width//4, -height//3),
                  (3*width//4, height//3), (0, 100, 255, 255), -1)

    # Windows
    cv2.rectangle(car_img, (width//3, -height//4),
                  (2*width//3, height//4), (200, 230, 255, 255), -1)

    # Wheels
    wheel_color = (30, 30, 30, 255)
    wheel_size = (width//4, height//4)
    cv2.ellipse(car_img, (width//4, height),
                wheel_size, 0, 0, 360, wheel_color, -1)
    cv2.ellipse(car_img, (3*width//4, height),
                wheel_size, 0, 0, 360, wheel_color, -1)

    return car_img

# Improved hand gesture detection


def detect_hand_gesture(hand_landmarks):

    # Get landmark positions
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y])

    # Thumb, Index, Middle, Ring, Pinky
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]  # Previous joints

    fingers_up = []

    # For right hand: tip should be to the right of pip
    # For left hand: tip should be to the left of pip
    thumb_tip_x = landmarks[tip_ids[0]][0]
    thumb_pip_x = landmarks[pip_ids[0]][0]

    wrist_x = landmarks[0][0]
    thumb_tip_dist = abs(thumb_tip_x - wrist_x)
    thumb_pip_dist = abs(thumb_pip_x - wrist_x)

    if thumb_tip_dist > thumb_pip_dist:
        fingers_up.append(1)
    else:
        fingers_up.append(0)

    # Check other four fingers (compare y coordinates)
    # Tip should be above (lower y value) than pip joint for finger to be up
    for i in range(1, 5):
        tip_y = landmarks[tip_ids[i]][1]
        pip_y = landmarks[pip_ids[i]][1]

        # Some tolerance for detection
        if tip_y < pip_y - 0.02:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    # Count extended fingers
    total_fingers = sum(fingers_up)

    # More lenient thresholds for better detection
    if total_fingers >= 3:  # 3 or more fingers up = open palm
        return 'palm'
    elif total_fingers <= 1:  # 0-1 fingers up = fist
        return 'fist'
    else:  # 2 fingers = treat as fist for now
        return 'fist'


# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening camera feed")
    exit()

# Camera frame dimensions
width, height = 1000, 800
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create road background
road_img = create_road_background(width, height)

# Create car image
car_width, car_height = 80, 60
car_img = create_car_image(car_width, car_height)

# Car position
car_x = width // 2 - car_width // 2
car_y = height - car_height - 20
car_speed = 0
car_direction = 0  # -1 for left, 0 for straight, 1 for right
max_speed = 8
turn_speed = 5

# Game state
score = 0
obstacles = []
obstacle_timer = 0
game_over = False

# Calculate road boundaries
road_width = width // 2
road_left = (width - road_width) // 2
road_right = road_left + road_width

use_finger_count = True

print("Controls:")
print("- Both palms open: Forward")
print("- Both fists: Backward")
print("- Left palm + Right fist: Turn Left")
print("- Right palm + Left fist: Turn Right")
print("- Press 'Q' to quit")
print("- Press 'R' to restart when game over")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb)

    # Create game frame with road background
    game_frame = road_img.copy()

    # Draw score
    cv2.putText(game_frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Initialize hand states
    left_hand_state = "none"
    right_hand_state = "none"
    gesture_text = "Stop"

    # Process hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get hand type (left or right)
            hand_type = handedness.classification[0].label

            # Draw landmarks on game frame
            mp_drawing.draw_landmarks(
                game_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Detect gesture using finger count method
            hand_state = detect_hand_gesture(hand_landmarks)

            # Set hand color based on state
            if hand_state == "palm":
                hand_color = (0, 255, 0)  # Green for palm
            else:
                hand_color = (0, 0, 255)  # Red for fist

            # Update hand state variables
            if hand_type == "Left":
                left_hand_state = hand_state
            else:
                right_hand_state = hand_state

            # Draw hand state indicator with finger count for debugging
            wrist = hand_landmarks.landmark[0]
            x, y = int(wrist.x * width), int(wrist.y * height)

            # Get finger count for debugging
            landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            tip_ids = [4, 8, 12, 16, 20]
            pip_ids = [3, 6, 10, 14, 18]
            fingers_up = []

            # Thumb check
            wrist_x = landmarks[0][0]
            thumb_tip_dist = abs(landmarks[tip_ids[0]][0] - wrist_x)
            thumb_pip_dist = abs(landmarks[pip_ids[0]][0] - wrist_x)
            fingers_up.append(1 if thumb_tip_dist > thumb_pip_dist else 0)

            # Other fingers
            for i in range(1, 5):
                if landmarks[tip_ids[i]][1] < landmarks[pip_ids[i]][1] - 0.02:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            finger_count = sum(fingers_up)
            cv2.putText(game_frame, f"{hand_type}: {hand_state} ({finger_count})",
                        (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

    # Determine movement based on hand states
    if not game_over:
        if left_hand_state == "fist" and right_hand_state == "fist":
            # Both fists: backward
            car_speed = max(-max_speed, car_speed - 1)
            gesture_text = "Backward"
        elif left_hand_state == "palm" and right_hand_state == "palm":
            # Both palms: forward
            car_speed = min(max_speed, car_speed + 1)
            gesture_text = "Forward"
        elif left_hand_state == "palm" and right_hand_state == "fist":
            # Left palm, right fist: left turn
            car_direction = max(-1, car_direction - 0.2)
            gesture_text = "Turn Left"
        elif left_hand_state == "fist" and right_hand_state == "palm":
            # Left fist, right palm: right turn
            car_direction = min(1, car_direction + 0.2)
            gesture_text = "Turn Right"
        else:
            # Slow down if no specific gesture
            if car_speed > 0:
                car_speed = max(0, car_speed - 0.5)
            elif car_speed < 0:
                car_speed = min(0, car_speed + 0.5)
            gesture_text = "Stop"
    else:
        gesture_text = "GAME OVER"

    # Update car position
    car_y -= car_speed
    car_x += car_direction * turn_speed

    # Convert to integers to avoid OpenCV errors
    car_x = int(car_x)
    car_y = int(car_y)

    # Boundary checks and game over detection
    if car_x < road_left or car_x + car_width > road_right:
        # Car touched road border - game over
        game_over = True

    if car_y < 0:
        car_y = 0
    elif car_y > height - car_height:
        car_y = height - car_height

    # Add friction to direction
    car_direction *= 0.9
    if abs(car_direction) < 0.1:
        car_direction = 0

    # Draw gesture text
    cv2.putText(game_frame, gesture_text, (width // 2 - 100, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Draw speed indicator
    cv2.putText(game_frame, f"Speed: {abs(car_speed)}", (width - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Game logic (only when moving forward)
    if not game_over and car_speed > 0:
        # Add to score based on speed
        score += int(car_speed)

        # Create obstacles
        obstacle_timer += 1
        if obstacle_timer > 30 - min(25, car_speed * 2):
            obstacle_timer = 0
            # Add new obstacle
            obstacle_width = np.random.randint(40, 80)
            obstacle_x = np.random.randint(
                width // 4, 3 * width // 4 - obstacle_width)
            obstacles.append({
                'x': obstacle_x,
                'y': -50,
                'width': obstacle_width,
                'height': np.random.randint(30, 60),
                'speed': np.random.randint(3, 7)
            })

    # Update and draw obstacles
    for obstacle in obstacles[:]:
        obstacle['y'] += obstacle['speed']

        # Draw obstacle
        cv2.rectangle(game_frame,
                      (obstacle['x'], obstacle['y']),
                      (obstacle['x'] + obstacle['width'],
                       obstacle['y'] + obstacle['height']),
                      (200, 50, 50), -1)

        # Collision detection
        if (car_x < obstacle['x'] + obstacle['width'] and
            car_x + car_width > obstacle['x'] and
            car_y < obstacle['y'] + obstacle['height'] and
                car_y + car_height > obstacle['y']):
            game_over = True

        # Remove obstacles that are off screen
        if obstacle['y'] > height:
            obstacles.remove(obstacle)

    # Draw car (simplified - just a rectangle for now to avoid RGBA issues)
    cv2.rectangle(game_frame, (car_x, car_y), (car_x + car_width,
                  car_y + car_height), (0, 100, 255), -1)
    cv2.rectangle(game_frame, (car_x + car_width//4, car_y - 10),
                  (car_x + 3*car_width//4, car_y + 20), (0, 100, 255), -1)
    cv2.rectangle(game_frame, (car_x + car_width//3, car_y - 5),
                  (car_x + 2*car_width//3, car_y + 10), (200, 230, 255), -1)

    # Draw wheels
    cv2.circle(game_frame, (car_x + car_width//4,
               car_y + car_height), 8, (30, 30, 30), -1)
    cv2.circle(game_frame, (car_x + 3*car_width//4,
               car_y + car_height), 8, (30, 30, 30), -1)

    # Draw game over message
    if game_over:
        cv2.putText(game_frame, "GAME OVER! Press 'R' to restart",
                    (width // 2 - 250, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(game_frame, f"Final Score: {score}",
                    (width // 2 - 120, height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display game frame
    cv2.imshow("Hand Gesture Driving Game", game_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and game_over:
        # Reset game
        game_over = False
        car_x = width // 2 - car_width // 2
        car_y = height - car_height - 20
        car_speed = 0
        car_direction = 0
        obstacles = []
        score = 0

# Clean up
hands.close()
cap.release()
cv2.destroyAllWindows()
