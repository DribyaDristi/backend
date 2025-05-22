from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model("newmodel/Dribya_Dristi.h5")

def ef(image):
    img = load_img(image, color_mode="grayscale", target_size=(48, 48))
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0

label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def identify_all_images(test_folder):
    results = []

    # Get all image files from the test folder
    image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in sorted(image_files):
        image_path = os.path.join(test_folder, image_file)

        # Process the image
        img = ef(image_path)

        # Show preprocessed image
        plt.imshow(img[0, :, :, 0], cmap='gray')
        plt.title(f"Preprocessed: {image_file}")
        plt.axis('off')
        plt.show()

        # Prediction
        pred = model.predict(img, verbose=0)
        pred_label = label[pred.argmax()]

        # Get true label from filename (remove '_test.jpg')
        true_label = image_file.split('_')[0].upper()

        # Store results
        results.append({
            'Image': image_file,
            'Predicted': pred_label,
            'Actual': true_label,
            'Correct': pred_label == true_label
        })

        print(f"Image: {image_file:<20} Predicted: {pred_label:<10} Actual: {true_label:<10}")

    # Create a DataFrame and calculate accuracy
    df = pd.DataFrame(results)
    accuracy = (df['Correct'].sum() / len(df)) * 100

    print("\n" + "="*50)
    print(f"Total images processed: {len(df)}")
    print(f"Overall accuracy: {accuracy:.2f}%")

    # Save results to CSV
    csv_path = "asl_test_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")

if __name__ == "__main__":
    test_folder = "asl_alphabet_test/asl_alphabet_test"
    identify_all_images(test_folder)