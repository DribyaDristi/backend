import cv2
from cvzone import HandTrackingModule as htm
# Initialize the hand detector
detector = htm.HandDetector(detectionCon=0.8, maxHands=1)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)  # Detect hands in the image
    # hands, img = detector.findPosition(img, draw=False)  # Get hand positions
    cv2.imshow("Video Feed", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()