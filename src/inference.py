import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model as model_manager
import pid

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_image(image_path):
    """
    do inference for a given image and display the result and the residual
    """
    original_image = Image.open(image_path)  # load the image from a given path
    residual = pid.apply_pid_algorithm(original_image, train=False)  # EDIT: deterministic center crop for inference
    display_image = original_image.resize((256, 256))  # resize to match the size of the residual for display

    residual_tensor = torch.from_numpy(residual).permute(2, 0, 1).float()  # convert to tensor and permute the color channels
    residual_tensor = residual_tensor.unsqueeze(0).to(DEVICE)  # specify the batch size to be 1 and move to gpu

    # EDIT: use the best-validation checkpoint for inference, not just the
    # latest saved shard (which could be an overfit late-training state).
    try:
        model = model_manager.load_best_model(DEVICE)
    except FileNotFoundError:
        print("[WARN] No best_model.pth found yet -- falling back to latest checkpoint.")
        model, _, _, _ = model_manager.get_model(DEVICE)
    model.eval()  # set the model to evaluation mode

    with torch.no_grad():  # disable gradient computation
        outputs = model(residual_tensor)  # forward pass
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()  # calculate softmax out of the logitsof the final layer and move the result to cpu
        predicted = np.argmax(probabilities)  # get the class with the highest probability and call it the predicted class

    class_label = "AI-Generated" if predicted == 1 else "Human-Generated"  # print based on the predicted class
    confidence = probabilities[predicted] * 100
    print(f"[INFERENCE RESULT] The model thinks this image is {confidence:.2f}% {class_label}.")

    # display the original and residual images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image (256x256)")
    plt.imshow(display_image)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Residual Image (256x256)")
    plt.imshow(residual)
    plt.axis("off")
    plt.show()

    # return display_image , residual , predicted


if __name__ == "__main__":
    # Example usage
    image_path = "F:/projects/PiD_AIGID/images/real_1.jpg"  # Replace with the actual image path
    infer_image(image_path)
