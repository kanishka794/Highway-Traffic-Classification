import torch
import torch.nn as nn
import timm
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import tempfile

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
# Image Transform (IMPORTANT FIX)
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
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
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]

    flow_x = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
    flow_y = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = np.stack([flow_x, flow_y, flow_x], axis=2).astype(np.uint8)

    return flow_rgb


# ---------------------------
# Convert Video to MP4
# ---------------------------
def convert_to_mp4(input_path):

    cap = cv2.VideoCapture(input_path)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_file.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    return output_path


# ---------------------------
# Prediction Function
# ---------------------------
def predict_video(video):

    video = convert_to_mp4(video)

    cap = cv2.VideoCapture(video)

    ret, prev_frame = cap.read()

    if not ret:
        return str(video), "Could not read video"

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)

    probabilities = []

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Sample every 10th frame
        if frame_count % 10 != 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        flow = compute_optical_flow(prev_frame, frame)

        rgb_tensor = transform(Image.fromarray(frame))
        flow_tensor = transform(Image.fromarray(flow))

        combined = torch.cat([rgb_tensor, flow_tensor], dim=0)

        combined = combined.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(combined)

            probs = torch.softmax(output, dim=1)

        probs = probs.cpu().numpy()[0]

        probabilities.append(probs)

        # Debug print
        print("Frame Probabilities:", probs)

        prev_frame = frame

    cap.release()

    if len(probabilities) == 0:
        return str(video), "No frames processed"

    probabilities = np.array(probabilities)

    avg_probs = np.mean(probabilities, axis=0)

    pred_class = int(np.argmax(avg_probs))

    final_prediction = CLASS_NAMES[pred_class]

    confidence = float(avg_probs[pred_class])*100

    result = f"Traffic Condition: {final_prediction} | Confidence: {confidence:.2f}"

    return str(video), result








# ---------------------------
# Helper: Get Video Info
# ---------------------------
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = frame_count / fps if fps != 0 else 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return fps, duration, width, height







# ---------------------------
# Prediction Wrapper
# ---------------------------
def run_prediction(video):
    if video is None:
        return None, "❌ Please upload a video first", "<div>Waiting...</div>", ""

    video_path, result = predict_video(video)

    parts = result.split("|")
    label = parts[0].split(":")[1].strip()
    confidence = parts[1].split(":")[1].strip()

    color_map = {
        "Light": "#2ecc71",
        "Medium": "#f39c12",
        "Heavy": "#e74c3c"
    }

    color = color_map.get(label, "#7f8c8d")

    # ✨ Animated Prediction Card
    html = f"""
    <div class="traffic-card" style="
        background: {color};
        padding:25px;
        border-radius:12px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
        box-shadow:0px 6px 18px rgba(0,0,0,0.2);
        animation: fadeIn 0.8s ease-in-out;">
        
        🚦 {label}
        <br><br>
        📊 Confidence: {confidence}
    </div>

    <style>
    .traffic-card {{
        color: white;
    }}

    .traffic-info {{
        background: #ffffff !important;
        color: #111111 !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1) !important;
        border: 1px solid #dddddd !important;
    }}

    body[data-theme="dark"] .traffic-info,
    .gradio-container.dark .traffic-info,
    .gradio-dark .traffic-info {{
        background: #1f2227 !important;
        color: #f5f5f5 !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.45) !important;
        border: 1px solid #444a58 !important;
    }}

    body[data-theme="dark"] .traffic-info b,
    .gradio-container.dark .traffic-info b,
    .gradio-dark .traffic-info b {{
        color: #f8f8f8 !important;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """

    # 🎥 Video Info Card
    fps, duration, w, h = get_video_info(video_path)

    info_html = f"""
    <div class="traffic-info" style="
        margin-top:15px;
        padding:15px;
        border-radius:10px;
        animation: fadeIn 1s ease-in-out;">

        <b>📄 Video Details</b><br><br>
        ⏱ Duration: {duration:.2f} sec<br>
        🎞 FPS: {fps:.2f}<br>
        📐 Resolution: {w} × {h}
    </div>
    """

    return video_path, "✅ Analysis Complete", html, info_html


# ---------------------------
# UI Layout
# ---------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
<div style="text-align:center;">
    <h1>🚗 Highway Traffic Classification</h1>
    <p style="color:gray; font-size:16px;">
        AI-based Traffic Detection System
    </p>
</div>
""")

    with gr.Row():

        # LEFT SIDE
        with gr.Column():

            video_input = gr.Video(label="Upload Video", interactive=True)

            status_text = gr.Textbox(
                label="Status",
                value="🟡 Waiting for video...",
                interactive=False
            )

            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyse", variant="primary")
                reset_btn = gr.Button("♻️ Reset")

            video_info = gr.HTML("")

        # RIGHT SIDE
        with gr.Column():

            video_output = gr.Video(label="Processed Video")

            output_html = gr.HTML("<div>No result yet</div>")

    # ---------------------------
    # Actions
    # ---------------------------
    analyze_btn.click(
        fn=run_prediction,
        inputs=video_input,
        outputs=[video_output, status_text, output_html, video_info]
    )

    reset_btn.click(
        fn=lambda: (None, "🟡 Waiting for video...", "<div>No result</div>", ""),
        inputs=[],
        outputs=[video_input, status_text, output_html, video_info]
    )

# Launch
demo.launch(share=True)