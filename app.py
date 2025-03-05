import streamlit as st
import cv2
import numpy as np
import easyocr
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from ultralytics import YOLO
from datetime import datetime
import os

os.environ["OMP_NUM_THREADS"] = "1"



# Google Sheets Setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("Vehicle Tracker").sheet1

# Load YOLO Model (License Plate Detection)
model = YOLO("yolov8n.pt")

# OCR Reader (EasyOCR handles plates well)
reader = easyocr.Reader(['en'])

# Detect Plate and Extract Text
def extract_plate(image):
    # Run YOLO detection
    results = model.predict(image, conf=0.5)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = image[y1:y2, x1:x2]

            # OCR on the detected plate region
            plate_text = reader.readtext(plate_img, detail=0)
            if plate_text:
                return plate_text[0], plate_img

    return None, None

# Streamlit Interface
st.title("🚗 Advanced Vehicle Number Plate Tracker")

option = st.radio("Choose an option:", ["📸 Capture Photo", "📤 Upload Image"])
image = None

if option == "📸 Capture Photo":
    uploaded_file = st.camera_input("Capture number plate")
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

elif option == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload a number plate image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

# Process and Extract Plate
if image is not None:
    st.image(image, caption="Uploaded Image", use_column_width=True)

    plate_number, plate_crop = extract_plate(image)

    if plate_number:
        st.success(f"✅ Detected Number Plate: {plate_number}")
        st.image(plate_crop, caption="Detected Plate", use_column_width=False)

        checkpoint = st.text_input("Enter checkpoint (e.g., Point A, Point B)")
        if st.button("Submit"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet.append_row([plate_number, checkpoint, timestamp])
            st.write(f"✅ Data saved: {plate_number} at {timestamp}")
    else:
        st.error("❌ No plate detected. Try again!")
