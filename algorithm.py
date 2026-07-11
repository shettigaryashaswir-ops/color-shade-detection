import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import requests
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Color Finder Pro", layout="wide") # Wide layout for side-by-side view
st.title("🎨 Color Finder Pro")
st.write("Click anywhere on your image to instantly identify the shade and see a live comparison.")

# ==========================================
# 2. LOAD COLOR DATASET
# ==========================================
CSV_URL = "https://raw.githubusercontent.com/codebrainz/color-names/master/output/colors.csv"
CSV_FILE = "colors.csv"

@st.cache_data
def load_dataset():
    column_names = ["color_name", "hex", "R", "G", "B"]
    
    if not os.path.exists(CSV_FILE):
        try:
            response = requests.get(CSV_URL, timeout=10)
            with open(CSV_FILE, "wb") as f:
                f.write(response.content)
        except Exception:
            # Emergency fallback if internet drops
            return pd.DataFrame([
                ["Pure Black", "#000000", 0, 0, 0], 
                ["Pure White", "#ffffff", 255, 255, 255]
            ], columns=column_names)
            
    # FIXED: Always read with header=None and explicit names to support raw unheadered CSVs perfectly
    return pd.read_csv(CSV_FILE, names=column_names, header=None)

color_df = load_dataset()

# ==========================================
# 3. WEIGHTED RGB MATCHING ENGINE
# ==========================================
def find_nearest_shade(r, g, b):
    # Humans see Green best, then Red, then Blue. We weight the math to match our eyes!
    r_weight, g_weight, b_weight = 0.30, 0.59, 0.11
    
    distances = np.sqrt(
        r_weight * (color_df['R'] - r) ** 2 + 
        g_weight * (color_df['G'] - g) ** 2 + 
        b_weight * (color_df['B'] - b) ** 2
    )
    
    match_idx = distances.idxmin()
    return color_df.loc[match_idx]

# ==========================================
# 4. SIDEBAR INPUTS
# ==========================================
mode = st.sidebar.radio("Image Source:", ("📤 Upload Picture", "📷 Use Webcam"))
target_image = None

if mode == "📤 Upload Picture":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        target_image = Image.open(uploaded_file).convert("RGB")
else:
    webcam_file = st.camera_input("Take a snapshot")
    if webcam_file is not None:
        target_image = Image.open(webcam_file).convert("RGB")

# ==========================================
# 5. SIDE-BY-SIDE INTERACTIVE LAYOUT
# ==========================================
if target_image is not None:
    # Split the screen: 55% left for image, 45% right for results
    col_left, col_right = st.columns([1.2, 1.0], gap="large")
    
    with col_left:
        st.write("### 🎯 Step 1: Click a color below")
        target_image.thumbnail((600, 600))
        img_width, img_height = target_image.size
        
        # Capture the mouse click position
        click_data = streamlit_image_coordinates(target_image, width=img_width, key=f"picker_{mode}")
    
    with col_right:
        st.write("### 📊 Step 2: View Results")
        if click_data is not None:
            x, y = click_data["x"], click_data["y"]
            
            if x < img_width and y < img_height:
                # Extract RGB of the clicked pixel
                img_array = np.array(target_image)
                r, g, b = img_array[y, x]
                hex_clicked = f"#{r:02x}{g:02x}{b:02x}"
                
                # Find the closest matching row in the spreadsheet
                matched_row = find_nearest_shade(r, g, b)
                hex_matched = matched_row['hex']
                
                # --- VISUAL COLOR CARD ---
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 1px solid #ddd; box-shadow: 2px 2px 12px rgba(0,0,0,0.05);">
                    <h2 style="color: #333; margin-top: 0;">{matched_row['color_name']}</h2>
                    <hr style="margin: 10px 0; border: 0; border-top: 1px solid #ccc;">
                    
                    <p style="margin-bottom: 5px; font-weight: bold; color: #555;">Color Comparison:</p>
                    <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                        <div style="flex: 1; text-align: center;">
                            <div style="background-color: {hex_clicked}; height: 60px; border-radius: 6px; border: 1px solid #aaa;"></div>
                            <span style="font-size: 0.8em; color: #666;">Clicked Pixel</span>
                        </div>
                        <div style="flex: 1; text-align: center;">
                            <div style="background-color: {hex_matched}; height: 60px; border-radius: 6px; border: 1px solid #aaa;"></div>
                            <span style="font-size: 0.8em; color: #666;">Closest Shade</span>
                        </div>
                    </div>
                    
                    <table style="width:100%; font-size: 0.9em; color: #444; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #eee;"><td style="padding: 5px 0;"><b>Your RGB:</b></td><td style="text-align: right;">[{r}, {g}, {b}]</td></tr>
                        <tr style="border-bottom: 1px solid #eee;"><td style="padding: 5px 0;"><b>Match RGB:</b></td><td style="text-align: right;">[{matched_row['R']}, {matched_row['G']}, {matched_row['B']}]</td></tr>
                        <tr style="border-bottom: 1px solid #eee;"><td style="padding: 5px 0;"><b>Your HEX:</b></td><td style="text-align: right;"><code>{hex_clicked}</code></td></tr>
                        <tr><td style="padding: 5px 0;"><b>Match HEX:</b></td><td style="text-align: right;"><code>{hex_matched}</code></td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click anywhere on the image to the left to display its color profile card here!")