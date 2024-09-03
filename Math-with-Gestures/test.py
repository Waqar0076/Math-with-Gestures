import cv2
import numpy as np
from PIL import Image
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai

# Streamlit configuration
st.set_page_config(page_title="Math with Gestures", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        margin: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 18px;
        margin-bottom: 10px;
    }
    .output {
        margin-top: 20px;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar configuration
st.sidebar.image('MathGestures.png', use_column_width=True)
st.sidebar.title("Math with Gestures")
st.sidebar.markdown("Control the app using the options below:")
run = st.sidebar.checkbox('Run', value=True)

# Google Generative AI configuration
genai.configure(api_key="API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1600)  # Increased width for larger display
cap.set(4, 900)   # Increased height for larger display

# Initialize the HandDetector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Layout configuration
st.header("Math with Gestures")
st.subheader("Real-Time Analysis")
st.markdown("---")

col1, col2 = st.columns([4, 1])  # Adjusted column width for larger display

with col1:
    FRAME_WINDOW = st.empty()

with col2:
    output_text_area = st.empty()

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None, img

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""

prev_pos = None
canvas = None
output_text = ""

while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to capture image from camera")
        break

    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.markdown(f"**Output:** {output_text}")

    cv2.waitKey(1)
