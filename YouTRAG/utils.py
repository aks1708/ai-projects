from youtube_transcript_api import YouTubeTranscriptApi
from pytubefix import YouTube

import whisper

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.readers.file import PyMuPDFReader

from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_index.core.postprocessor import LLMRerank

from fpdf import FPDF

# ============= PDF Creator =============

def write_text_to_pdf(text, output_pdf):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0,5, text)
    pdf.output(output_pdf)

# ============= Configurations =============
Settings.embed_model = HuggingFaceEmbedding(model_name="infly/inf-retriever-v1-1.5b")
Settings.llm = Ollama(model="gemma3:1b", temperature=0)

rerank = LLMRerank(llm=Settings.llm, top_n=3)

whisper_model = whisper.load_model("base")
# ============= Audio and Transcript Processing =============

def download_audio(video_url):
    yt = YouTube(video_url)
    audio = yt.streams.filter(only_audio=True).first()
    audio.download(filename="audio.mp3")
    return "audio.mp3"

def get_transcription(video_url):
    """
    Get the transcript of a YouTube video.
    
    Args:
        video_url (str): URL of the YouTube video
        
    Returns:
        str: Transcript of the video
    """
    yt = YouTube(video_url)
    try:
        try:
            transcript_list = YouTubeTranscriptApi().list_transcripts(yt.video_id)
            preferred_languages = ['en', 'en-US', 'en-GB', 'en-IN']
            transcript_finder = transcript_list.find_manually_created_transcript(preferred_languages)
            lang_code = list(filter(lambda x: x.code.startswith(transcript_finder.language_code), yt.captions))[0].code
            transcript = yt.captions[lang_code].generate_txt_captions()
        except Exception as e:
            transcript = whisper_model.transcribe(download_audio(video_url))['text']
        
        write_text_to_pdf(transcript, f"youtube/{yt.title}.pdf")
        return f'youtube/{yt.title}.pdf'

    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")

# ============= Vector Store and Chat Engine Setup =============
def setup_chat_engine(transcript_file):
    """
    Set up vector store and create a chat engine.
    
    Args:
        documents: Documents to be embedded and stored in Pinecone
        
    Returns:
        ChatEngine: Configured chat engine
    """
    client = QdrantClient(host="localhost", port=6333)
    if client.collection_exists("youtube"):
        client.delete_collection("youtube")

    vector_store = QdrantVectorStore(client=client, 
    collection_name="youtube", 
    enable_hybrid=True,
    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions")
    
    # Load and process documents
    reader = PyMuPDFReader()
    documents = reader.load_data(transcript_file)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    retriever = index.as_retriever(similarity_top_k=5)
    query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[rerank])

    # Configure chat engine
    memory = ChatMemoryBuffer.from_defaults()
    chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    verbose=False,
    query_engine=query_engine)

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