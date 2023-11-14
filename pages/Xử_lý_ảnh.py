import streamlit as st
import time
import numpy as np
import ImageProcessing.Chapter3 as c3
import ImageProcessing.Chapter4 as c4
import ImageProcessing.Chapter5 as c5
import ImageProcessing.Chapter9 as c9
import cv2
import object_detection as od
from PIL import Image

st.set_page_config(page_title="Xử lý ảnh số", page_icon="🌅")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-042.jpg");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)
st.markdown("# Xử lý ảnh")
st.write()

image_file = st._main.file_uploader("Upload Your Image", type=[
                                  'jpg', 'png', 'jpeg', 'tif'])

global imgin
if image_file is not None:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) 

    chapter_options = ["Chapter 3", "Chapter 4", "Chapter 5", "Chapter 9"]
    selected_chapter = st.sidebar.selectbox("Select an option", chapter_options)


    if selected_chapter == "Chapter 3":
        chapter3_options = ["Negative", "Logarit", "Power", "PiecewiseLinear", "Histogram", "HistogramEqualization",
                                                "HistogramEqualizationColor", "LocalHistogram", "HistogramStatistics", 
                                                "BoxFilter", "GaussFilter","Threshold", "MedianFilter", "Sharpen", "Gradient"]
        
        chapter3_selected = st.sidebar.selectbox("Select an option", chapter3_options)    
        if chapter3_selected  == "Negative":
            processed_image = c3.Negative(imgin)
        elif chapter3_selected  == "Logarit":
            processed_image = c3.Logarit(imgin)
        elif chapter3_selected  == "Power":
            processed_image = c3.Power(imgin)
        elif chapter3_selected  == "PiecewiseLinear":
            processed_image = c3.PiecewiseLinear(imgin)
        elif chapter3_selected  == "Histogram":
            processed_image = c3.Histogram(imgin)
        elif chapter3_selected  == "HistogramEqualization":
            processed_image = c3.HistEqual(imgin)
        elif chapter3_selected == "HistogramEqualizationColor":
            imgin = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            processed_image = c3.HistEqualColor(imgin)
        elif chapter3_selected  == "LocalHistogram":
            processed_image = c3.LocalHist(imgin)
        elif chapter3_selected  == "HistogramStatistics":
            processed_image = c3.HistStat(imgin)
        elif chapter3_selected  == "BoxFilter":
            processed_image = c3.BoxFilter(imgin)
        elif chapter3_selected  == "GaussFilter":
            processed_image = c3.GaussFilter(imgin)
        elif chapter3_selected  == "Threshold":
            processed_image = c3.Threshold(imgin)
        elif chapter3_selected  == "MedianFilter":
            processed_image = c3.MedianFilter(imgin)
        elif chapter3_selected  == "Sharpen":
            processed_image = c3.Sharpen(imgin)
        elif chapter3_selected  == "Gradient":
            processed_image = c3.Gradient(imgin) 
            
                      
    elif selected_chapter == "Chapter 4":
        
        chapter4_options = ["Spectrum", "FrequencyFilter", "DrawFilter", "RemoveMoire"]
        
        chapter4_selected = st.sidebar.selectbox("Select an option", chapter4_options)   
        
        
        if chapter4_selected == "Spectrum":
            processed_image = c4.Spectrum(imgin)
        elif chapter4_selected == "FrequencyFilter":
            processed_image = c4.FrequencyFilter(imgin)
        elif chapter4_selected == "DrawFilter":
            imgin = Image.new('RGB', (5, 5),  st.get_option("theme.backgroundColor"))
            processed_image = c4.DrawNotchRejectFilter()
        elif chapter4_selected == "RemoveMoire":
            processed_image = c4.RemoveMoire(imgin)
            

    elif selected_chapter == "Chapter 5":
        
        chapter5_options = ["CreateMotionNoise", "DenoiseMotion", "DenoisestMotion"]
        chapter5_selected = st.sidebar.selectbox("Select an option", chapter5_options)   
        
        if chapter5_selected == "CreateMotionNoise":
            processed_image = c5.CreateMotionNoise(imgin)
        elif chapter5_selected == "DenoiseMotion":
            processed_image = c5.DenoiseMotion(imgin)
        elif chapter5_selected == "DenoisestMotion":
            temp = cv2.medianBlur(imgin, 7)
            processed_image = c5.DenoiseMotion(temp)   
            
                 
    elif selected_chapter == "Chapter 9":
        
        chapter9_options = ["ConnectedComponent", "CountRice"]
        chapter9_selected = st.sidebar.selectbox("Select an option", chapter9_options)   
        
        if chapter9_selected  == "ConnectedComponent":
            processed_image = c9.ConnectedComponent(imgin)    
        elif chapter9_selected  == "CountRice":
            processed_image = c9.CountRice(imgin)
    

    st.subheader("Original Image and Processed Image")
    st.image([imgin, processed_image], width = 350)
st.button("Re-run")
