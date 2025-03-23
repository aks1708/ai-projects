from pytubefix import YouTube
import os
import whisper
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.ollama import OllamaEmbedding

from dotenv import load_dotenv


SUMMARISATION_PROMPT = """
"Generate a summary of the following transcript.\n"
"Transcript:"
{text}
"""

# ============= Environment Setup =============
load_dotenv()

# ============= LLM and Embedding Configuration =============
Settings.embed_model = OllamaEmbedding("mxbai-embed-large")
Settings.llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key=os.environ.get("GOOGLE_API_KEY"))

# ============= Audio and Transcript Processing =============
    
def download_audio(video_url):
    """
    Download audio from a YouTube video URL.

    Args:
        video_url (str): The URL of the YouTube video to download

    Returns:
        tuple: A tuple containing:
            - YouTube object: The YouTube object containing video metadata
            - str: The filepath where the audio was saved
    """
    yt = YouTube(video_url)
    filename = os.path.join('youtube', f"{yt.title}.mp4")
    yt.streams.get_audio_only().download(output_path='youtube', filename=f"{yt.title}.mp4")
    return yt, filename

def transcribe_audio(audio_file):
    """
    Transcribe audio from an MP4 file using Whisper model and save the transcript.

    Args:
        audio_file (str): Path to the MP4 audio file to transcribe

    Returns:
        str: Path to the generated transcript text file
    """
    filepath = audio_file.replace(".mp4", ".txt")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    with open(filepath, "w") as f:
        f.write(result["text"])
    return filepath

# ============= Text Processing and Summarization =============
def summarise_transcript(transcript_text):
    """
    Generate a summary of the transcript using the configured LLM.
    
    Args:
        transcript_text (str): Path to the transcript file
        
    Returns:
        str: Generated summary
    """
    with open(transcript_text, 'r') as f:
        text = f.read()
    
    response = Settings.llm.complete(SUMMARISATION_PROMPT.format(text=text))
    return response.text

# ============= Vector Store and Chat Engine Setup =============
def setup_chroma(transcript_file):
    """
    Set up ChromaDB vector store and create a chat engine.
    
    Args:
        documents: Documents to be embedded and stored in ChromaDB
        
    Returns:
        ChatEngine: Configured chat engine
    """
    # Clear cache and create new collection
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    chroma_client = chromadb.Client()
    
    # Create or reset collection
    if "video" in chroma_client.list_collections():
        chroma_client.delete_collection("video")
    chroma_collection = chroma_client.create_collection("video")
    
    # Load and process documents
    documents = SimpleDirectoryReader(input_files=[transcript_file]).load_data()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Configure chat engine
    memory = ChatMemoryBuffer.from_defaults()
    chat_engine = index.as_chat_engine(
        chat_mode="best",
        memory=memory,
        verbose=False)

    return chat_engine

# ============= Query Processing =============
def query(query_str, chat_engine):
    """
    Process a query using the chat engine.
    
    Args:
        query_str (str): User's query
        chat_engine (ChatEngine): Configured chat engine
        
    Returns:
        response (str): Response from the chat engine
    """
    # Process with the chat engine
    response = chat_engine.chat(query_str)
    return response.response

