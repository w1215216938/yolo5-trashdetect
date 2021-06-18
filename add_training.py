import streamlit as st
output_path = 'runs/detect/exp/1.mp4'
video_file = open(output_path, 'rb')
video_bytes = video_file.read()
st.video(video_bytes)


