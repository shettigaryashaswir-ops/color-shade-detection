import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import requests
from streamlit_image_coordinates import streamlit_image_coordinates
from skimage.color import rgb2lab, deltaE_ciede2000

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Color Finder Pro", layout="wide") # Wide layout for side-by-side view
st.title("🎨 Color Finder Pro")
st.write("Click anywhere on your image to instantly identify the shade using industrial-grade CIEDE2000 accuracy.")

# ==========================================
# 2. LOAD COLOR DATASET
# ==========================================
CSV_URL = "https://githubusercontent.com"
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
            
    # Always read with header=None and explicit names to support raw unheadered CSVs perfectly
    return pd.read_csv(CSV_FILE, names=column_names, header=None)

color_df = load_dataset()

# ==========================================
# 3. INDUSTRIAL HIGH-ACCURACY MATCHING ENGINE (FIXED)
# ==========================================
def find_nearest_shade(r, g, b):
    # 1. Normalize clicked RGB pixel to 0.0 - 1.0 range
    clicked_rgb = np.array([r / 255.0, g / 255.0, b / 255.0])
    
    # 2. Reshape pixel to 3D matrix shape (1 row, 1 column, 3 channels)
    clicked_rgb_3d = clicked_rgb.reshape((1, 1, 3))
    clicked_lab = rgb2lab(clicked_rgb_3d)
    
    # 3. Pull R, G, B columns explicitly from your dataset
    r_vals = color_df['R'].values / 255.0
    g_vals = color_df['G'].values / 255.0
    b_vals = color_df['B'].values / 255.0
    
    # 4. Create a 3D matrix for the dataset to match the library format perfectly
    # Shape becomes (1 row, N colors, 3 channels)
    dataset_rgb_3d = np.stack([r_vals, g_vals, b_vals], axis=-1).reshape((1, -1, 3))
    dataset_lab = rgb2lab(dataset_rgb_3d)
    
    # 5. Calculate Delta E 2000 across the perfectly aligned dimensions
    deltas = deltaE_ciede2000(clicked_lab, dataset_lab)
    
    # 6. Flatten to match the DataFrame sequence
    deltas_flat = deltas.flatten()
    match_idx = np.argmin(deltas_flat)
    min_delta = deltas_flat[match_idx]
    
    # 7. Convert Delta E cleanly to an intuitive 0-100% accuracy score
    accuracy_score = max(0, int((1 - (min_delta / 100.0)) * 100))
    
    return color_df.iloc[match_idx], accuracy_score



# ==========================================
# 4. SIDEBAR INPUTS & ACCURACY CONTROLS
# ==========================================
mode = st.sidebar.radio("Image Source:", ("📤 Upload Picture", "📷 Use Webcam"))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Match Strictness Settings")

# User sets their preferred strictness threshold
strictness_threshold = st.sidebar.slider(
    "Minimum Required Accuracy (%)", 
    min_value=50, 
    max_value=100, 
    value=85,
    help="Higher percentage means the app will reject loose color name matches."
)

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
                
                # Find the closest matching row and calculate match accuracy via CIEDE2000
                matched_row, accuracy_score = find_nearest_shade(r, g, b)
                
                # Check if it meets the user's strictness setting
                if accuracy_score >= strictness_threshold:
                    # --- VISUAL COLOR CARD ---
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 1px solid #ddd; box-shadow: 2px 2px 12px rgba(0,0,0,0.05);">
                        <h2 style="color: #333; margin-top: 0; margin-bottom: 10px;">{matched_row['color_name']}</h2>
                        <span style="background-color: #e1f5fe; color: #0288d1; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: bold;">
                            🎯 Accuracy Match: {accuracy_score}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ App found a name but its accuracy ({accuracy_score}%) is lower than your set threshold ({strictness_threshold}%). Try clicking a different spot or lower your strictness bar in the sidebar.")
        else:
            st.info("Click anywhere on the image to the left to display its color profile card here!")
