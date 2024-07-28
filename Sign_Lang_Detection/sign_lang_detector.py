import cv2
from transformers import pipeline
from cvzone.HandTrackingModule import HandDetector

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Load a pre-trained image classification model from Hugging Face
model = pipeline('image-classification', model='google/vit-base-patch16-224')

# Function to preprocess the hand region for the model
def preprocess_hand(hand_img):
    hand_img = cv2.resize(hand_img, (224, 224))  # Resize to the model's expected input size
    hand_img = hand_img / 255.0  # Normalize pixel values
    return hand_img

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Find hand in the frame
    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]  # Only take the first hand detected
        x, y, w, h = hand['bbox']  # Get the bounding box
        hand_img = frame[y:y+h, x:x+w]  # Crop the hand image

        # Preprocess the hand image
        hand_img = preprocess_hand(hand_img)

        # Make prediction using the model
        predictions = model(hand_img)

        # Get the predicted class
        predicted_class = predictions[0]['label']

        # Print the recognized gesture to the console
        print(f'Recognized Gesture: {predicted_class}')

        # Display the recognized gesture on the frame
        cv2.putText(frame, f'Gesture: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
