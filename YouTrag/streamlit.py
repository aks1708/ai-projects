import streamlit as st
import os
import utils
import time
from pytubefix import YouTube

# Set page config
st.set_page_config(
    page_title="YouTRAG",
    layout="wide"
)

# Custom CSS for blue sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #000080;
    }
    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #000000;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #333333;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    [data-testid="stSidebar"] .stTextInput input {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Create necessary directories if they don't exist
os.makedirs('youtube', exist_ok=True)

def stream_data(text: str, delay: float = 0.02):
    """Stream text word by word with a specified delay."""
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# Sidebar for video input
with st.sidebar:
    st.header("🎥 YouTRAG")
    st.markdown('Get key insights from YouTube talks and podcasts ')
    video_url = st.text_input("Enter YouTube URL", key="video_url")
    
    if st.button("Upload"):
        if video_url:
            try:
                # Create a progress bar for the overall process
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get video title and check for existing transcript
                yt = YouTube(video_url)
                transcript_file = os.path.join('youtube', f"{yt.title}.txt")
                
                if os.path.exists(transcript_file):
                    status_text.markdown("<p style='color: white;'>Using existing transcript...</p>", unsafe_allow_html=True)
                    progress_bar.progress(25)
                    st.success(f"Found existing transcript for: {yt.title}", icon="✅")
                else:
                    # Step 1: Download audio
                    status_text.markdown("<p style='color: white;'>Downloading audio...</p>", unsafe_allow_html=True)
                    progress_bar.progress(25)
                    yt, filename = utils.download_audio(video_url)
                    st.success(f"Downloaded: {yt.title}", icon="✅")
                    
                    # Step 2: Transcribe audio
                    status_text.markdown("<p style='color: white;'>Transcribing audio...</p>", unsafe_allow_html=True)
                    transcript_file = utils.transcribe_audio(filename)
                    progress_bar.progress(50)
                    st.success("Transcription complete!", icon="✅")
                    os.remove(filename)  # Clean up the audio file
                
                # Step 3: Generate summary
                status_text.markdown("<p style='color: white;'>Generating summary...</p>", unsafe_allow_html=True)
                summary = utils.summarise_transcript(transcript_file)
                progress_bar.progress(75)
                st.success("Summary generated!", icon="✅")
                
                # Step 4: Setup chat engine
                status_text.markdown("<p style='color: white;'>Setting up chat engine...</p>", unsafe_allow_html=True)
                chat_engine = utils.setup_chroma(transcript_file)
                st.session_state['chat_engine'] = chat_engine
                st.session_state['transcript_file'] = transcript_file
                st.session_state['summary'] = summary
                st.session_state['video_title'] = yt.title
                progress_bar.progress(100)
                status_text.markdown("<p style='color: green;'>Complete!</p>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a YouTube URL first")
    
    # Add reset chat button that only appears when chat engine is set up
    if 'chat_engine' in st.session_state:
        if st.button("Reset Chat", key="reset_chat"):
            # Reset chat engine using the stored transcript file
            st.session_state['chat_engine'] = utils.setup_chroma(st.session_state['transcript_file'])
            # Clear chat messages
            st.session_state.messages = []
            st.rerun()

# Main content area
if 'summary' in st.session_state:
    # Display video title as header
    st.header(f"{st.session_state.get('video_title', 'Video Title')}")
    
    # Embed YouTube video
    if 'video_url' in st.session_state:
        video_id = st.session_state.video_url.split('v=')[1].split('&')[0]
        st.markdown(f"""
        <iframe 
            width="100%" 
            height="400" 
            src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
        """, unsafe_allow_html=True)
    
    # Display summary
    st.subheader("📝 Summary")
    st.write(st.session_state['summary'])
   
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = utils.query(prompt, st.session_state.chat_engine)
        st.write_stream(stream_data(response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})