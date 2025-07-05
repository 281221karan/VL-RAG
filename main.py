import os
import json
import streamlit as st
import openai
import tempfile
import torch
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
import gc
from openai import OpenAI
from IPython.display import display
from google import genai
import time
os.environ['HF_HOME'] = "/teamspace/studios/this_studio/huggingface_cache"

from functions import (
    load_images,
    load_retrieval_model,
    load_vlm_model,
    get_query,
    retrieval,
    augmentation_and_generation,
)

# ✅ Streamlit setup
st.set_page_config(
    page_title="GPT-4o Chat",
    page_icon="💬",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tmp_path" not in st.session_state:
    st.session_state.tmp_path = None
if "images" not in st.session_state:
    st.session_state.images = None

# ✅ Custom top-left layout
st.markdown("""
    <style>
        .main { padding-top: 1rem !important; padding-left: 1rem !important; }
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# ✅ Sidebar: API key input + uploader
st.sidebar.title("🔑 Gemini API Key")

api_key = st.sidebar.text_input(
    "Enter your Google Gemini API key:",
    type="password",
    placeholder="Paste your API key here..."
)

st.sidebar.markdown(
    "[Get a free Gemini API key](https://aistudio.google.com/apikey)",
    unsafe_allow_html=True
)

st.sidebar.title("📁 Upload File")
uploaded_file = st.sidebar.file_uploader(
    label="Upload a file",
    type=["pdf", "txt", "csv", "jpg", "png"]
)

# ──────────────────────────────────────
# ✅ Chat Title
st.markdown("""
    <div style='
        font-size: 28px;
        font-weight: 600;
        margin-top: -90px;
        margin-bottom: 20px;
        text-align: left;
        font-family: sans-serif;
    '>
        🤖 Vision Language RAG - ChatBot
    </div>
""", unsafe_allow_html=True)

# ✅ Show status message if no file is uploaded
if not uploaded_file:
    st.info("📢 No file uploaded — you're chatting with Google's Gemini model.")

# ✅ Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Function to render chat bubbles
def render_chat(role, content):
    if role == "user":
        # ✅ User message in a right-aligned bubble
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #2e2e2e; color: white; padding: 10px 15px;
                            border-radius: 12px; max-width: 70%; margin-bottom: 10px;
                            font-family: sans-serif;'>
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # ❌ Assistant response: plain markdown (left-aligned)
        st.markdown(content)

# ✅ Display previous messages
for message in st.session_state.chat_history:
    render_chat(message["role"], message["content"])

# ✅ Chat input field
user_prompt = st.chat_input("Ask VL-RAG...")

if uploaded_file and st.session_state.tmp_path is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.session_state.tmp_path = tmp_file.name

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    render_chat("user", user_prompt)

    # Show "Thinking..." bubble
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
        <div style='display: flex; justify-content: flex-start;'>
            <div style='background-color: #F1F0F0; color: grey; padding: 10px 15px;
                        border-radius: 12px; max-width: 70%; margin-bottom: 10px;
                        font-family: sans-serif; font-style: italic;'>
                Thinking, the backend GPU is a bit slow it might take a few seconds.....
            </div>
        </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        if st.session_state.images is None:
            if st.session_state.tmp_path:
                st.session_state.images = load_images(st.session_state.tmp_path)
            else:
                st.error("❌ Images not available. Please re-upload the file.")
                st.stop()
        images = st.session_state.images

        # 🔁 RAG pipeline
        retrieval_model, retrieval_processor = load_retrieval_model(
            retrieval_model_name="vidore/colqwen2.5-v0.2",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None
        )
        
        vlm_model, vlm_processor = load_vlm_model(
            vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        if st.session_state.tmp_path:
            images = load_images(st.session_state.tmp_path)
        else:
            st.error("❌ No document path found. Please re-upload.")
            st.stop()

        # ✅ Ensure images are loaded (handles page reloads)
        if st.session_state.images is None:
            if st.session_state.tmp_path:
                st.session_state.images = load_images(st.session_state.tmp_path)
            else:
                st.error("❌ Images not available. Please re-upload the file.")
                st.stop()

        images = st.session_state.images
        
        topk_scores, topk_index = retrieval(
            k=3,
            queries=[user_prompt],
            images=images,
            retrieval_model=retrieval_model,
            retrieval_processor=retrieval_processor,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        output_text = augmentation_and_generation(
            queries=[user_prompt],
            images=images,
            topk_scores=topk_scores,
            topk_index=topk_index,
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu"
        )[0]

    else:
        # 🔁 Gemini fallback
        if not api_key:
            output_text = "❌ Please enter your Google Gemini API key to chat."
        else:
            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt
                )
                output_text = response.text
            except Exception as e:
                output_text = f"❌ Error from Gemini API: {str(e)}"

    # ✅ Final assistant response
    thinking_placeholder.empty()
    st.session_state.chat_history.append({"role": "assistant", "content": output_text})
    assistant_placeholder = st.empty()
    typed_text = ""
    for char in output_text:
        typed_text += char
        assistant_placeholder.markdown(typed_text)
        time.sleep(0.004)  # Typing speed (you can tweak this)