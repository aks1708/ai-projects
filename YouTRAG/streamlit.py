import torch
torch.classes.__path__ = []

import streamlit as st
import os
import utils
from pytubefix import YouTube

# Set page config
st.set_page_config(
    page_title="YouTRAG",
    layout="wide"
)

# Create necessary directories if they don't exist
os.makedirs('youtube', exist_ok=True)

# Sidebar for video input
with st.sidebar:
    st.header("🎥 YouTRAG")
    st.markdown('Get key insights from YouTube videos')
    video_url = st.text_input("Enter YouTube URL", key="video_url")
    
    if st.button("Upload"):
        if video_url:
            try:
                yt = YouTube(video_url)
                # Create a progress bar for the overall process
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get video title and check for existing transcript
                st.session_state['video_title'] = yt.title
                st.session_state['transcript_file'] = os.path.join('youtube', f"{yt.title}.pdf")
                
                if os.path.exists(st.session_state['transcript_file']):
                    progress_bar.progress(50)  
                    status_text.markdown(f"Found existing transcript for:\n{yt.title}")
                else:  
                    #Transcribe audio
                    status_text.markdown(f"Transcribing the video")
                    st.session_state['transcript_file'] = utils.get_transcription(video_url)
                    progress_bar.progress(50)
                    status_text.markdown(f"Transcription done for the video:\n{yt.title}")
                    
                    if os.path.exists("audio.mp3"):
                        os.remove("audio.mp3")
                
                # Step 3: Setup chat engine
                status_text.markdown("Setting up chat engine...")
                st.session_state['chat_engine'] = utils.setup_chat_engine(st.session_state['transcript_file'])
                st.session_state.messages = []
                progress_bar.progress(100)
                st.success("Fire away!", icon="🚀")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a YouTube URL first")
    

if "messages" not in st.session_state:
    st.session_state.messages = []

# Add reset chat button that only appears when chat engine is set up
if 'chat_engine' in st.session_state:
    
    st.header(st.session_state['video_title'])
    # Embed YouTube video
    if 'video_url' in st.session_state:
        st.video(st.session_state['video_url'])
    
    with st.sidebar:
        if st.button("Reset Chat"):
            # Reset chat engine using the stored transcript file
            with st.spinner("Resetting the conversation..."):
                st.session_state['chat_engine'].reset()
                st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = utils.query(prompt, st.session_state['chat_engine'])
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})