import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from model import Sim2RealBackbone
import time

MODEL_PATH = "./saved_models/best_scratch_model.pth"
THRESHOLD = 0.42 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    print("Loading Sim2Real Engine...")
    model = Sim2RealBackbone(pretrained=False)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Error: best_model.pth not found. Train first!")
        exit()
        
    model.to(DEVICE)
    model.eval() 
    return model

def get_embedding(model, face_img):
    trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = trans(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        emb = model(tensor)
        emb = torch.nn.functional.normalize(emb)
    return emb

def main():
    model = load_model()
    mtcnn = MTCNN(keep_all=False, min_face_size=60, device=DEVICE)
    cap = cv2.VideoCapture(0)

    database_emb = None
    user_name = "Unknown"
    state = "IDLE" 

    print("\n" + "="*30)
    print("   SIM2REAL LIVE DEMO")
    print("="*30)
    print("Press 'E' -> ENROLL (Takes a snapshot of you)")
    print("Press 'Q' -> QUIT")
    print("="*30 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0: continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if state == "ENROLL_NOW":
                    database_emb = get_embedding(model, face_crop)
                    user_name = "User"
                    state = "VERIFYING"
                    print("--> USER ENROLLED! Scanning for match...")

                elif state == "VERIFYING" and database_emb is not None:
                    live_emb = get_embedding(model, face_crop)
                    
                    score = torch.sum(live_emb * database_emb).item()
                    
                    if score > THRESHOLD:
                        color = (0, 255, 0)
                        label = f"MATCH ({score:.2f})"
                    else:
                        color = (0, 0, 255)
                        label = f"IMPOSTER ({score:.2f})"
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        status_text = "Press 'E' to Enroll your Face" if state == "IDLE" else "Scanning..."
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Sim2Real Security System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('e'): state = "ENROLL_NOW"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()