from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
import cv2
import numpy as np
from hand_detector import get_hand_crop
from classify import classify_image
import time

class ASLDetectorApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # Camera preview
        self.image = Image()
        self.label = Label(text='', size_hint=(1, .1))
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.label)

        # Initialize variables for stability detection
        self.previous_positions = []
        self.stability_threshold = 60
        self.stability_frames = 20
        self.last_prediction_time = 0
        self.prediction_cooldown = 3
        self.last_prediction = None
        self.prediction_display_duration = 3

        # Setup video capture
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)
        
        return self.layout

    def calculate_hand_center(self, cropped_hand):
        if cropped_hand is None:
            return None
        h, w = cropped_hand.shape[:2]
        return (w//2, h//2)

    def is_hand_stable(self, positions, threshold):
        if len(positions) < self.stability_frames:
            return False
        
        max_distance = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                             (positions[i][1] - positions[j][1])**2)
                max_distance = max(max_distance, dist)
        
        return max_distance < threshold

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            current_time = time.time()
            
            # Process frame
            cropped_hand, output_frame = get_hand_crop(frame)
            
            if cropped_hand is not None:
                hand_center = self.calculate_hand_center(cropped_hand)
                if hand_center:
                    self.previous_positions.append(hand_center)
                    self.previous_positions = self.previous_positions[-self.stability_frames:]
                    
                    if (self.is_hand_stable(self.previous_positions, self.stability_threshold) and 
                        current_time - self.last_prediction_time > self.prediction_cooldown):
                        try:
                            predicted_sign = classify_image(cropped_hand)
                            self.last_prediction = (predicted_sign, current_time)
                            self.last_prediction_time = current_time
                        except Exception as e:
                            print("Prediction error:", e)
                    else:
                        cv2.putText(output_frame, "Stabilizing...", (10, 120), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display prediction
            if self.last_prediction and (current_time - self.last_prediction[1] <= self.prediction_display_duration):
                self.label.text = f"Predicted Sign: {self.last_prediction[0]}"
                cv2.putText(output_frame, self.last_prediction[0], (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Convert frame to texture for Kivy
            buf = cv2.flip(output_frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    ASLDetectorApp().run()