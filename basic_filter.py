from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np

def apply_blur_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))
        img_blurred.save("blurred_image.jpg")
        print("Processed image saved as 'blurred_image.jpg'.")
    except Exception as e:
        print(f"Error processing image: {e}")

def apply_edge_detection(image_path):
    try:
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        img_edges = img.filter(ImageFilter.FIND_EDGES)
        img_edges.save("edges_image.jpg")
        print("Processed image saved as 'edges_image.jpg'.")
    except Exception as e:
        print(f"Error processing image: {e}")

def apply_sharpen_filter(image_path):
    try:
        img = Image.open(image_path)
        img_sharpened = img.filter(ImageFilter.SHARPEN)
        img_sharpened.save("sharpened_image.jpg")
        print("Processed image saved as 'sharpened_image.jpg'.")
    except Exception as e:
        print(f"Error processing image: {e}")

def apply_emboss_filter(image_path):
    try:
        img = Image.open(image_path)
        img_embossed = img.filter(ImageFilter.EMBOSS)
        img_embossed.save("embossed_image.jpg")
        print("Processed image saved as 'embossed_image.jpg'.")
    except Exception as e:
        print(f"Error processing image: {e}")

def apply_rust_tint(image_path, rust_texture_path):
    try:
        img = Image.open(image_path).convert("RGB")
        rust_texture = Image.open(rust_texture_path).convert("RGB")
        rust_texture = rust_texture.resize(img.size)  # Match image size

        blended = Image.blend(img, rust_texture, alpha=0.3)  # Adjust tint intensity
        blended.save("rust_tinted_image.jpg", "JPEG")
        print("Processed image saved as 'rust_tinted_image.jpg'.")
    except Exception as e:
        print(f"Error processing image: {e}")

def apply_pencil_sketch(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img_inverted = cv2.bitwise_not(img)  # Invert image
        img_blurred = cv2.GaussianBlur(img_inverted, (21, 21), sigmaX=0, sigmaY=0)  # Blur
        img_blend = cv2.divide(img, 255 - img_blurred, scale=256)  # Blend for sketch effect
        cv2.imwrite("pencil_sketch.jpg", img_blend)
        print("Processed image saved as 'pencil_sketch.jpg'.")
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    image_path = "Parrot.jpg"  # Replace with the path to your image file

    #apply_blur_filter(image_path)
    #apply_edge_detection(image_path)
    #apply_sharpen_filter(image_path)
    #apply_emboss_filter(image_path)
    apply_pencil_sketch(image_path)

