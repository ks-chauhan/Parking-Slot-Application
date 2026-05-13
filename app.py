import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
import os

from parking_model import predict_slot

# -----------------------------------
# Streamlit Config
# -----------------------------------

st.set_page_config(
    page_title="Smart Parking System",
    layout="wide"
)

st.title("🚗 Smart Parking System")

# -----------------------------------
# Upload Parking Image
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Parking Lot Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # -----------------------------------
    # Load Image
    # -----------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    # Save uploaded image
    image.save("uploaded_parking_image.jpg")

    # Original dimensions
    original_width, original_height = image.size

    # -----------------------------------
    # Resize Image For Display
    # -----------------------------------

    DISPLAY_WIDTH = 1000

    scale_ratio = DISPLAY_WIDTH / original_width

    display_height = int(
        original_height * scale_ratio
    )

    display_image = image.resize(
        (DISPLAY_WIDTH, display_height)
    ).convert("RGBA")

    display_np = np.array(display_image)

    # -----------------------------------
    # Slot Annotation Section
    # -----------------------------------

    st.subheader("Draw Parking Slots")

    st.markdown("""
    Instructions:
    - Draw rectangles around parking slots
    - Each rectangle represents one parking space
    """)

    # -----------------------------------
    # Drawable Canvas
    # -----------------------------------

    canvas_result = st_canvas(
        fill_color="rgba(0,255,0,0.2)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=display_image,
        update_streamlit=True,
        height=display_np.shape[0],
        width=display_np.shape[1],
        drawing_mode="rect",
        key="canvas",
    )

    # -----------------------------------
    # Extract Parking Slots
    # -----------------------------------

    parking_slots = []

    if canvas_result.json_data is not None:

        objects = canvas_result.json_data["objects"]

        for obj in objects:

            x = int(obj["left"] / scale_ratio)
            y = int(obj["top"] / scale_ratio)
            w = int(obj["width"] / scale_ratio)
            h = int(obj["height"] / scale_ratio)

            slot = {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h
            }

            parking_slots.append(slot)

    # -----------------------------------
    # Display Slot Coordinates
    # -----------------------------------

    st.subheader("Detected Parking Slots")

    st.write(parking_slots)

    # -----------------------------------
    # Save Parking Layout
    # -----------------------------------

    if st.button("Save Parking Layout"):

        with open("parking_slots.json", "w") as f:

            json.dump(
                parking_slots,
                f,
                indent=4
            )

        st.success(
            "Parking layout saved successfully!"
        )

    # -----------------------------------
    # Predict Button
    # -----------------------------------

    predict_button = st.button(
        "Predict Parking Occupancy"
    )

    # -----------------------------------
    # Run Inference
    # -----------------------------------

    if predict_button:

        # -----------------------------------
        # Load Saved Image
        # -----------------------------------

        image = Image.open(
            "uploaded_parking_image.jpg"
        ).convert("RGB")

        preview_image = np.array(image).copy()

        # -----------------------------------
        # Load Parking Slots
        # -----------------------------------

        if os.path.exists("parking_slots.json"):

            with open(
                "parking_slots.json",
                "r"
            ) as f:

                parking_slots = json.load(f)

        else:

            st.error(
                "No parking layout found."
            )

            st.stop()

        # -----------------------------------
        # Occupancy Detection
        # -----------------------------------

        occupied_count = 0

        for i, slot in enumerate(parking_slots):

            sx1 = slot["x1"]
            sy1 = slot["y1"]
            sx2 = slot["x2"]
            sy2 = slot["y2"]

            # -----------------------------------
            # Crop Parking Slot
            # -----------------------------------

            slot_crop = preview_image[
                sy1:sy2,
                sx1:sx2
            ]

            # Prevent invalid crops
            if slot_crop.size == 0:
                continue

            # -----------------------------------
            # AI Prediction
            # -----------------------------------

            prediction = predict_slot(
                slot_crop
            )

            # -----------------------------------
            # Determine Slot Status
            # -----------------------------------

            if prediction == "Occupied":

                color = (0, 0, 255)

                occupied_count += 1

            else:

                color = (0, 255, 0)

            # -----------------------------------
            # Draw Slot Rectangle
            # -----------------------------------

            cv2.rectangle(
                preview_image,
                (sx1, sy1),
                (sx2, sy2),
                color,
                3
            )

            # -----------------------------------
            # Draw Slot Label
            # -----------------------------------

            cv2.putText(
                preview_image,
                f"{prediction}",
                (sx1, sy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # -----------------------------------
        # Statistics
        # -----------------------------------

        total_slots = len(parking_slots)

        available_slots = (
            total_slots - occupied_count
        )

        st.subheader("Parking Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Slots",
            total_slots
        )

        col2.metric(
            "Occupied",
            occupied_count
        )

        col3.metric(
            "Available",
            available_slots
        )

        # -----------------------------------
        # Final Result
        # -----------------------------------

        st.subheader(
            "AI Parking Occupancy Result"
        )

        st.image(
            preview_image,
            channels="RGB"
        )
