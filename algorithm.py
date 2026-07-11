import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from skimage.color import rgb2lab, deltaE_ciede2000

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Color Finder Pro", layout="wide") 
st.title("🎨 Color Finder Pro")
st.write("Click anywhere on your image to instantly identify the shade in simple, normal terms.")

# ==========================================
# 2. DEFINING UNIVERSALLY UNDERSTOOD SHADES
# ==========================================
@st.cache_data
def load_dataset():
    # A cleanly curated database of standard, recognizable color shades
    normal_colors = [
        # Red spectrum
        ["Pure Red", 255, 0, 0], ["Crimson Red", 153, 0, 0], ["Dark Red", 139, 0, 0],
        ["Cherry Red", 210, 4, 45], ["Ruby Red", 224, 17, 95], ["Wine Red / Burgundy", 128, 0, 32],
        
        # Pink spectrum
        ["Light Pink", 255, 182, 193], ["Hot Pink", 255, 105, 180], ["Baby Pink", 244, 194, 194],
        ["Rose Pink", 255, 102, 204], ["Salmon Pink", 255, 145, 164], ["Coral Pink", 248, 131, 121],
        
        # Orange spectrum
        ["Pure Orange", 255, 165, 0], ["Dark Orange", 255, 140, 0], ["Peach", 255, 218, 185],
        ["Rust / Terracotta", 184, 115, 51], ["Tangerine", 242, 133, 0], ["Coral Orange", 255, 127, 80],
        
        # Yellow & Brown spectrum
        ["Pure Yellow", 255, 255, 0], ["Light Yellow", 255, 255, 224], ["Mustard Yellow", 225, 173, 1],
        ["Gold", 255, 215, 0], ["Cream / Ivory", 255, 253, 208], ["Beige", 245, 245, 220],
        ["Light Brown", 181, 101, 29], ["Chocolate Brown", 123, 63, 0], ["Dark Brown", 101, 67, 33],
        ["Coffee Brown", 111, 78, 55], ["Tan / Khaki", 210, 180, 140], ["Caramel", 198, 115, 39],
        
        # Green spectrum
        ["Pure Green", 0, 255, 0], ["Dark Green", 0, 100, 0], ["Light Green", 144, 238, 144],
        ["Lime Green", 50, 205, 50], ["Olive Green", 128, 128, 0], ["Forest Green", 34, 139, 34],
        ["Mint Green", 152, 251, 152], ["Sage Green", 143, 151, 121], ["Emerald Green", 80, 200, 120],
        ["Teal / Cyan", 0, 128, 128], ["Seafoam Green", 159, 226, 191], ["Army Green", 75, 83, 32],
        
        # Blue spectrum
        ["Pure Blue", 0, 0, 255], ["Dark Blue", 0, 0, 139], ["Navy Blue", 0, 0, 128],
        ["Light Blue", 173, 216, 230], ["Sky Blue", 135, 206, 235], ["Baby Blue", 137, 207, 240],
        ["Royal Blue", 65, 105, 225], ["Turquoise / Aqua", 64, 224, 208], ["Midnight Blue", 25, 25, 112],
        ["Electric Blue", 125, 249, 255], ["Steel Blue", 70, 130, 180], ["Indigo", 63, 0, 255],
        
        # Purple & Violet spectrum
        ["Purple", 128, 0, 128], ["Violet", 238, 130, 238], ["Lavender", 230, 230, 250],
        ["Lilac", 200, 162, 200], ["Plum Purple", 221, 160, 221], ["Magenta / Fuchsia", 255, 0, 255],
        ["Mauve", 224, 176, 255], ["Grape Purple", 111, 45, 168], ["Eggplant Purple", 97, 64, 81],
        
        # Greyscale spectrum
        ["Pure White", 255, 255, 255], ["Off-White", 248, 249, 250], ["Light Grey", 211, 211, 211],
        ["Grey", 128, 128, 128], ["Dark Grey", 169, 169, 169], ["Charcoal Grey", 54, 69, 79],
        ["Silver", 192, 192, 192], ["Pure Black", 0, 0, 0]
    ]
    
    df = pd.DataFrame(normal_colors, columns=["color_name", "R", "G", "B"])
    return df

color_df = load_dataset()

# ==========================================
# 3. HIGH-ACCURACY CALCULATION ENGINE
# ==========================================
def find_nearest_shade(r, g, b):
    clicked_rgb = np.array([r / 255.0, g / 255.0, b / 255.0])
    clicked_rgb_3d = clicked_rgb.reshape((1, 1, 3))
    clicked_lab = rgb2lab(clicked_rgb_3d)
    
    r_vals = color_df['R'].values / 255.0
    g_vals = color_df['G'].values / 255.0
    b_vals = color_df['B'].values / 255.0
    
    dataset_rgb_3d = np.stack([r_vals, g_vals, b_vals], axis=-1).reshape((1, -1, 3))
    dataset_lab = rgb2lab(dataset_rgb_3d)
    
    deltas = deltaE_ciede2000(clicked_lab, dataset_lab)
    
    deltas_flat = deltas.flatten()
    match_idx = np.argmin(deltas_flat)
    min_delta = deltas_flat[match_idx]
    
    # Map strict Delta E values safely into a clean human percentage scale
    accuracy_score = max(0, int((1 - (min_delta / 60.0)) * 100))
    
    return color_df.iloc[match_idx], accuracy_score

# ==========================================
# 4. SIDEBAR INPUTS & ACCURACY CONTROLS
# ==========================================
mode = st.sidebar.radio("Image Source:", ("📤 Upload Picture", "📷 Use Webcam"))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Match Strictness Settings")

strictness_threshold = st.sidebar.slider(
    "Minimum Required Accuracy (%)", 
    min_value=40, 
    max_value=100, 
    value=65,
    help="Adjust if the app warns you that your shade match is too loose."
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
    col_left, col_right = st.columns([1.2, 1.0], gap="large")
    
    with col_left:
        st.write("### 🎯 Step 1: Click a color below")
        target_image.thumbnail((600, 600))
        img_width, img_height = target_image.size
        
        click_data = streamlit_image_coordinates(target_image, width=img_width, key=f"picker_{mode}")
    
    with col_right:
        st.write("### 📊 Step 2: View Results")
        if click_data is not None:
            x, y = click_data["x"], click_data["y"]
            
            if x < img_width and y < img_height:
                img_array = np.array(target_image)
                r, g, b = img_array[y, x]
                
                matched_row, accuracy_score = find_nearest_shade(r, g, b)
                
                if accuracy_score >= strictness_threshold:
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 1px solid #ddd; box-shadow: 2px 2px 12px rgba(0,0,0,0.05);">
                        <h2 style="color: #333; margin-top: 0; margin-bottom: 10px;">{matched_row['color_name']}</h2>
                        <span style="background-color: #e1f5fe; color: #0288d1; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: bold;">
                            🎯 Match Accuracy: {accuracy_score}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ Closest normal shade found is '{matched_row['color_name']}' but its accuracy match is low ({accuracy_score}%). Try lowering your threshold bar in the sidebar.")
        else:
            st.info("Click anywhere on the image to the left to see its shade name!")
