import torch
import torch.nn as nn
import timm
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import shutil
import tempfile
import os
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# ---------------------------
# Model Definition
# ---------------------------
class TrafficModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            in_chans=6,
            num_classes=0,
            global_pool="avg"
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )
 
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
 
 
# ---------------------------
# Load Model
# ---------------------------
model = TrafficModel().to(DEVICE)
checkpoint = torch.load("highway_traffic.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
CLASS_NAMES = checkpoint["class_names"]
print("Loaded Classes:", CLASS_NAMES)
model.eval()
 
 
# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
 
 
# ---------------------------
# Optical Flow
# ---------------------------
def compute_optical_flow(prev, curr):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_x = cv2.normalize(flow[:, :, 0], None, 0, 255, cv2.NORM_MINMAX)
    flow_y = cv2.normalize(flow[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = np.stack([flow_x, flow_y, flow_x], axis=2).astype(np.uint8)
    return flow_rgb
 
 
# ---------------------------
# Re-encode video to browser-compatible mp4
# ---------------------------
def reencode_to_mp4(input_path):
    """
    Re-encode the uploaded video to H.264 mp4 so any browser can play it.
    Returns path to the new temp file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    out_path = tmp.name
 
    cap = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
 
    cap.release()
    writer.release()
    return out_path
 
 
# ---------------------------
# Prediction Function
# ---------------------------
def predict_video(video):
    if video is None:
        return None, "No video uploaded."
 
    # Re-encode so the browser video player works
    playable_path = reencode_to_mp4(video)
 
    cap = cv2.VideoCapture(video)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return playable_path, "Could not read video."
 
    prev_frame  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    probs_list  = []
    frame_count = 0
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 != 0:
            continue
 
        frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flow     = compute_optical_flow(prev_frame, frame)
 
        rgb_t    = transform(Image.fromarray(frame))
        flow_t   = transform(Image.fromarray(flow))
        combined = torch.cat([rgb_t, flow_t], dim=0).unsqueeze(0).to(DEVICE)
 
        with torch.no_grad():
            probs = torch.softmax(model(combined), dim=1).cpu().numpy()[0]
 
        probs_list.append(probs)
        prev_frame = frame
 
    cap.release()
 
    if not probs_list:
        return playable_path, "No frames were processed."
 
    avg_probs  = np.mean(probs_list, axis=0)
    pred_idx   = int(np.argmax(avg_probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(avg_probs[pred_idx])
 
    # Build a readable result string
    lines = [
        f"Traffic Condition : {pred_class.upper()}",
        f"Confidence        : {confidence:.1%}",
        "",
        "All class probabilities:",
    ]
    for name, prob in zip(CLASS_NAMES, avg_probs):
        bar = "█" * int(prob * 30)
        lines.append(f"  {name:6s} {prob:.1%}  {bar}")
 
    return playable_path, "\n".join(lines)
 
 
# ---------------------------
# Gradio Interface
# ---------------------------
interface = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload Traffic Video (.avi / .mp4)"),
    outputs=[
        gr.Video(label="Video Preview"),
        gr.Textbox(label="Traffic Prediction", lines=8),
    ],
    title="Highway Traffic Classification",
    description=(
        "Upload a highway video clip. The model uses RGB frames + "
        "Farneback optical flow to classify traffic density as "
        "Light, Medium, or Heavy."
    ),
    flagging_mode="never",
)
 
interface.launch(share=True)