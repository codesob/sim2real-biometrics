import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from model import Sim2RealBackbone

PATH_BASELINE = "./saved_models/baseline_model.pth" 
PATH_PROPOSED = "./saved_models/best_model.pth"     

TEST_IMAGE_PATH = "./dataset/lfw_aligned/George_Bush/George_Bush_0001.jpg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_heatmap(model_path, input_tensor, rgb_img):
    """Loads a model and generates the Grad-CAM overlay"""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return rgb_img

    print(f"Loading {os.path.basename(model_path)}...")
    
    model = Sim2RealBackbone(pretrained=False).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    target_layers = [model.backbone.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        import glob
        print("Warning: Specific image not found. Searching for any jpg...")
        files = glob.glob("./dataset/lfw_aligned/*/*.jpg")
        if not files:
            print("CRITICAL: No LFW images found. Check path.")
            return
        image_path = files[0]
    else:
        image_path = TEST_IMAGE_PATH

    print(f"Testing Heatmap on: {image_path}")

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # BGR to RGB
    rgb_img = cv2.resize(rgb_img, (112, 112))
    rgb_float = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_float, 
                                   mean=[0.5, 0.5, 0.5], 
                                   std=[0.5, 0.5, 0.5]).to(DEVICE)

    vis_baseline = get_heatmap(PATH_BASELINE, input_tensor, rgb_float)

    vis_proposed = get_heatmap(PATH_PROPOSED, input_tensor, rgb_float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title("Input (Real Face)")
    axes[0].axis('off')

    axes[1].imshow(vis_baseline)
    axes[1].set_title("Baseline Attention\n(Texture Bias?)")
    axes[1].axis('off')

    axes[2].imshow(vis_proposed)
    axes[2].set_title("Proposed Attention\n(Shape Bias!)")
    axes[2].axis('off')

    plt.suptitle("Interpretability Analysis: Grad-CAM Feature Attention")
    save_path = "./final_heatmap_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison to {save_path}")

if __name__ == "__main__":
    main()