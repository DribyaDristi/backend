import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from hand_detector import get_hand_crop

# Load the trained model
model = tf.keras.models.load_model("newmodel/Dribya_Dristi.h5")

# Labels for classification
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            return None
            
        # Ensure image is in correct format
        if len(image.shape) == 2:
            # If grayscale, convert to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Resize to match training dimensions
        image = cv2.resize(image, (48, 48))
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Reshape and normalize
        image = image.reshape(1, 48, 48, 1)
        return image / 255.0
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Variables for prediction stability
    prediction_history = []
    history_length = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get hand crop using existing hand detector
        cropped_hand, annotated_frame = get_hand_crop(frame)
        
        if cropped_hand is not None:
            # Preprocess and predict
            # plt.imshow(cropped_hand)
            # plt.title(f"Cropped: Hand Image")
            # plt.axis('off')
            # plt.show()
            processed_img = preprocess_image(cropped_hand)
            # view processed image for debugging

            if processed_img is not None:
                plt.imshow(processed_img[0, :, :, 0], cmap='gray')
                plt.title(f"Preprocessed: Hand Image")
                plt.axis('off')
                plt.show()
                prediction = model.predict(processed_img, verbose=0)
                predicted_label = label[prediction.argmax()]
                
                # Add prediction to history
                prediction_history.append(predicted_label)
                if len(prediction_history) > history_length:
                    prediction_history.pop(0)
                
                # Get most common prediction from history
                if prediction_history:
                    stable_prediction = max(set(prediction_history), 
                                         key=prediction_history.count)
                    
                    # Draw prediction on frame
                    cv2.putText(annotated_frame, f"Sign: {stable_prediction}", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    print(f"Predicted Sign: {stable_prediction}")
                    
                    # Draw confidence bar
                    confidence = prediction.max() * 100
                    cv2.putText(annotated_frame, f"Confidence: {confidence:.1f}%",
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 2)
        # Display help text
        cv2.putText(annotated_frame, "Press 'q' to quit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow("ASL Sign Detection", annotated_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()