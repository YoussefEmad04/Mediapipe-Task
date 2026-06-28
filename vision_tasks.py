import cv2
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def detect_face(image_rgb):
    """
    Detect faces and draw bounding boxes.
    Input: RGB image
    Output: RGB image + message
    """
    output = image_rgb.copy()
    h, w, _ = output.shape

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:

        results = face_detection.process(image_rgb)

        if not results.detections:
            return output, "No face detected"

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            cv2.rectangle(
                output,
                (x, y),
                (x + box_w, y + box_h),
                (0, 255, 0),
                3
            )

        return output, f"Detected {len(results.detections)} face(s)"


def detect_hands(image_rgb):
    """
    Detect hands and draw 21 hand landmarks.
    MediaPipe Hands detects 21 3D landmarks for each hand.
    """
    output = image_rgb.copy()

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return output, "No hands detected"

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                output,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        return output, f"Detected {len(results.multi_hand_landmarks)} hand(s)"


def get_finger_states(hand_landmarks):
    """
    Simple rule-based finger counting.
    Returns list of opened/closed fingers:
    [thumb, index, middle, ring, pinky]
    """

    landmarks = hand_landmarks.landmark

    # Tips and PIP joints
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]

    fingers = []

    # Thumb: simple horizontal check
    # This works better for front-facing hand in many normal cases.
    thumb_is_open = landmarks[finger_tips[0]].x < landmarks[finger_pips[0]].x
    fingers.append(thumb_is_open)

    # Other fingers: tip higher than PIP means finger is open
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        finger_is_open = landmarks[tip].y < landmarks[pip].y
        fingers.append(finger_is_open)

    return fingers


def recognize_gesture_from_fingers(fingers):
    """
    Very simple gesture recognition from opened fingers.
    This is not a trained model. It is rule-based for learning.
    """

    opened_count = sum(fingers)

    if opened_count == 0:
        return "Fist"

    if opened_count == 5:
        return "Open Palm"

    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "Victory / Peace"

    if fingers[0] and not any(fingers[1:]):
        return "Thumbs Up / Like"

    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "Pointing"

    return f"{opened_count} finger(s) open"


def recognize_gesture(image_rgb):
    """
    Detect hand landmarks, then apply simple gesture rules.
    """
    output = image_rgb.copy()

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return output, "No hand detected"

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            output,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        fingers = get_finger_states(hand_landmarks)
        gesture = recognize_gesture_from_fingers(fingers)

        cv2.putText(
            output,
            gesture,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3
        )

        return output, f"Gesture: {gesture}"


def detect_pose(image_rgb):
    """
    Detect human body pose landmarks.
    """
    output = image_rgb.copy()

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:

        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return output, "No pose detected"

        mp_drawing.draw_landmarks(
            output,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        return output, "Pose detected"


def remove_background(image_rgb):
    """
    Segment person and blur the background.
    MediaPipe Selfie Segmentation is useful for selfie effects and video calls.
    """
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1
    ) as segmenter:

        results = segmenter.process(image_rgb)

        mask = results.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.5

        blurred_background = cv2.GaussianBlur(image_rgb, (55, 55), 0)

        output = np.where(condition, image_rgb, blurred_background)

        return output, "Background blurred / segmented"