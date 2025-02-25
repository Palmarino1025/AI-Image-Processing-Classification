import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load pre-trained model
model = MobileNetV2(weights="imagenet")

def load_and_preprocess_image(image_path):
    """Loads and preprocesses the image for MobileNetV2."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_array

def get_gradcam(model, img_array, class_idx):
    """Computes Grad-CAM heatmap for the predicted class."""
    
    # Get the last convolutional layer
    grad_model = tf.keras.models.Model(
        [model.input], 
        [model.get_layer("Conv_1").output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    heatmap = np.dot(conv_outputs, pooled_grads.numpy())

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def get_max_activation_region(heatmap, img_size):
    """Finds the most activated region from the heatmap."""
    heatmap_resized = cv2.resize(heatmap, img_size)
    threshold = 0.6 * np.max(heatmap_resized)  # Take the top 40% activation

    # Create binary mask
    mask = (heatmap_resized >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return (x, y, w, h)
    
    return None

def apply_occlusion(img, heatmap, occlusion_type="black"):
    """Applies occlusion based on heatmap activation region."""
    img_size = (img.size[0], img.size[1])
    occluded_img = np.array(img).copy()
    
    region = get_max_activation_region(heatmap, img_size)
    if not region:
        print("No high activation region found.")
        return img
    
    x, y, w, h = region

    if occlusion_type == "black":
        occluded_img[y:y+h, x:x+w] = 0  # Black box

    elif occlusion_type == "blur":
        blur = cv2.GaussianBlur(occluded_img[y:y+h, x:x+w], (15, 15), 0)
        occluded_img[y:y+h, x:x+w] = blur

    elif occlusion_type == "pixelation":
        small = cv2.resize(occluded_img[y:y+h, x:x+w], (10, 10), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        occluded_img[y:y+h, x:x+w] = pixelated

    return occluded_img

def overlay_heatmap(img, heatmap, alpha=0.5):
    """Overlays heatmap on the original image."""
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))  # Resize heatmap to image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.array(img)
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)  # Blend images
    return overlayed_img

def classify_image(model, img_array):
    """Classifies an image and returns the top prediction."""
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    
    class_label = decoded_predictions[0][1]  # Class Label
    confidence = decoded_predictions[0][2]  # Confidence Score
    
    return class_label, confidence

def classify_and_visualize(image_path):
    """Classifies image and applies Grad-CAM and occlusions."""
    
    img, img_array = load_and_preprocess_image(image_path)

    # Get original classification
    class_label, confidence = classify_image(model, img_array)
    print(f"Original Prediction: {class_label} ({confidence:.2f})")

    # Get Grad-CAM heatmap
    heatmap = get_gradcam(model, img_array, model.predict(img_array).argmax())

    # Overlay and save heatmap
    overlayed_img = overlay_heatmap(img, heatmap)
    cv2.imwrite("overlay_output.jpg", overlayed_img)
    print("Overlay image saved as overlay_output.jpg")

    # Apply occlusions and reclassify
    for occlusion_type in ["black", "blur", "pixelation"]:
        occluded_img = apply_occlusion(img, heatmap, occlusion_type)
        occluded_img_pil = image.array_to_img(occluded_img)
        
        # Preprocess occluded image
        occluded_array = image.img_to_array(occluded_img_pil)
        occluded_array = preprocess_input(occluded_array)
        occluded_array = np.expand_dims(occluded_array, axis=0)

        # Reclassify occluded image
        occluded_label, occluded_confidence = classify_image(model, occluded_array)
        print(f"{occlusion_type.capitalize()} Occlusion Prediction: {occluded_label} ({occluded_confidence:.2f})")

        # Save occluded image
        occlusion_filename = f"occluded_{occlusion_type}.jpg"
        cv2.imwrite(occlusion_filename, occluded_img)
        print(f"Occluded image saved as {occlusion_filename}")

if __name__ == "__main__":
    image_path = "Parrot.jpg"
    classify_and_visualize(image_path)
