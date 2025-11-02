import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import warnings
import os

warnings.filterwarnings("ignore")

# --- App Configuration ---
st.set_page_config(
    page_title="HF LLM Studio",
    page_icon="🤖",
    layout="wide",
)

# --- Device Selection ---
def get_device():
    """Detects and returns the best available device for PyTorch."""
    # Check for NVIDIA CUDA GPU
    if torch.cuda.is_available():
        st.sidebar.success("✅ NVIDIA GPU (CUDA) detected.")
        return torch.device("cuda")
    # Check for Apple Silicon M-series GPU
    if torch.backends.mps.is_available():
        st.sidebar.success("✅ Apple Silicon (MPS) detected.")
        return torch.device("mps")
    # Fallback to CPU
    st.sidebar.warning("⚠️ No GPU detected. Falling back to CPU. Performance may be slow.")
    return torch.device("cpu")

DEVICE = get_device()

# --- Model Loading ---
@st.cache_resource
def load_model(model_id):
    """Loads the LLM, tokenizer, and processor from Hugging Face."""

    # Read the token from environment variables
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    st.sidebar.info(f"Downloading and loading model: `{model_id}`...")
    if hf_token:
        st.sidebar.info("Using Hugging Face token.")

    # Configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    processor = None
    try:
        # Try loading a processor for multimodal models
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        st.sidebar.info("Model processor found (likely multimodal).")
    except Exception:
        st.sidebar.info("No processor found (likely a text-only model).")


    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map={"": DEVICE.type},
            token=hf_token
        )
        st.sidebar.success(f"✅ Model `{model_id}` loaded successfully!")
        return processor, tokenizer, model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.error(f"Could not load the model: {model_id}. Please check the model ID and your internet connection.")
        return None, None, None

# --- Main App Logic ---
st.title("🧠 Hugging Face LLM Studio")
st.markdown("A local playground to test and interact with open-source LLMs.")

# --- Sidebar ---
st.sidebar.header("Model Configuration")
DEFAULT_MODEL = "google/gemma-2b-it"

# Model selection input
model_id_input = st.sidebar.text_input(
    "Enter Hugging Face Model ID",
    DEFAULT_MODEL
)

st.sidebar.header("Generation Parameters")
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1,
    help="Controls randomness. Lower values are more deterministic, higher values are more creative."
)
max_new_tokens = st.sidebar.slider(
    "Max New Tokens",
    min_value=32,
    max_value=2048,
    value=512,
    step=64,
    help="The maximum number of tokens to generate in the response."
)

st.sidebar.header("Multimodal Input")
uploaded_image = st.sidebar.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Uploaded Image")

# Button to load the model
if st.sidebar.button("Load Model"):
    processor, tokenizer, model = load_model(model_id_input)
    # Store the loaded model info in session state to persist it
    st.session_state.processor = processor
    st.session_state.tokenizer = tokenizer
    st.session_state.model = model
    st.session_state.model_id = model_id_input
else:
    # If a model is already loaded in session state, use it.
    # Otherwise, load the default model on the first run.
    if 'model' not in st.session_state:
        processor, tokenizer, model = load_model(DEFAULT_MODEL)
        st.session_state.processor = processor
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.model_id = DEFAULT_MODEL
    else:
        processor = st.session_state.get("processor")
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model

# --- Chat Logic ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input
if prompt := st.chat_input("Ask me anything..."):
    # Store and display the user's message
    user_message = {"role": "user", "content": prompt}

    # Handle image if uploaded
    image_to_process = None
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image_to_process = image
        user_message["image"] = image # Store image with message

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        if "image" in user_message:
            st.image(user_message["image"], width=250)
        st.markdown(prompt)

    # Prepare for model response
    with st.chat_message("assistant"):
        model = st.session_state.get("model")
        tokenizer = st.session_state.get("tokenizer")
        processor = st.session_state.get("processor")

        if model is not None and tokenizer is not None:
            message_placeholder = st.empty()

            try:
                # Multimodal Input Processing
                if image_to_process and processor:
                    # LLaVA-like models expect a prompt like "USER: <image>\n<prompt>\nASSISTANT:"
                    prompt_template = f"USER: <image>\n{prompt}\nASSISTANT:"
                    inputs = processor(text=prompt_template, images=image_to_process, return_tensors="pt").to(DEVICE)
                # Text-only Input Processing
                else:
                    chat_template = [{"role": "user", "content": prompt}]
                    inputs_text = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(inputs_text, return_tensors="pt").to(DEVICE)

                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )

                # Decode and display response
                # We need to handle the decoding based on whether it was multi-modal or not
                # For simplicity, we decode the whole sequence and slice off the input part.
                # A more robust solution might be needed for some models.
                response_ids = outputs[0]
                full_response = tokenizer.decode(response_ids, skip_special_tokens=True)

                # For many models, the response includes the prompt, so we clean it up.
                # This is a basic way; more advanced cleanup might be needed.
                if processor and image_to_process:
                     # Find the start of the assistant's response
                    assistant_marker = "ASSISTANT:"
                    response_start = full_response.find(assistant_marker)
                    if response_start != -1:
                        full_response = full_response[response_start + len(assistant_marker):].strip()
                else:
                    # For text models using chat templates, the slicing is more reliable
                    input_token_len = inputs["input_ids"].shape[1]
                    response_only_ids = outputs[0][input_token_len:]
                    full_response = tokenizer.decode(response_only_ids, skip_special_tokens=True)


                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred during model generation: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Error during generation."})
        else:
            st.error("Model is not loaded. Cannot generate a response.")

    # Clear the uploaded image after processing
    # Note: This doesn't work perfectly in Streamlit's execution model.
    # The uploader widget state is managed by Streamlit.
    # A workaround is to process the image and then rely on the user to clear it.
