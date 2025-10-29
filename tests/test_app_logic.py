import pytest
from unittest.mock import patch, MagicMock
import torch
import streamlit as st

# Since we are testing the logic of the app, not Streamlit's execution,
# we need to import the functions directly.
from app import get_device, load_model

# To prevent Streamlit's UI elements from throwing errors during tests,
# we can mock the entire streamlit module.
st.sidebar = MagicMock()
st.error = MagicMock()

# --- Tests for get_device ---

@patch('torch.cuda.is_available', return_value=True)
def test_get_device_cuda(mock_cuda_available):
    """Test if get_device correctly returns 'cuda' when CUDA is available."""
    device = get_device()
    assert device.type == 'cuda'
    st.sidebar.success.assert_called_with("✅ NVIDIA GPU (CUDA) detected.")

@patch('torch.cuda.is_available', return_value=False)
@patch('torch.backends.mps.is_available', return_value=True)
def test_get_device_mps(mock_mps_available, mock_cuda_available):
    """Test if get_device correctly returns 'mps' when CUDA is not available but MPS is."""
    device = get_device()
    assert device.type == 'mps'
    st.sidebar.success.assert_called_with("✅ Apple Silicon (MPS) detected.")

@patch('torch.cuda.is_available', return_value=False)
@patch('torch.backends.mps.is_available', return_value=False)
def test_get_device_cpu(mock_mps_available, mock_cuda_available):
    """Test if get_device falls back to 'cpu' when no GPU is available."""
    device = get_device()
    assert device.type == 'cpu'
    st.sidebar.warning.assert_called_with("⚠️ No GPU detected. Falling back to CPU. Performance may be slow.")

# --- Tests for load_model (Lightweight) ---

@patch('app.AutoTokenizer.from_pretrained')
@patch('app.AutoModelForCausalLM.from_pretrained')
@patch('app.AutoProcessor.from_pretrained')
def test_load_model_success(mock_processor, mock_model, mock_tokenizer):
    """Test if load_model handles a successful model load."""
    # To test the function's logic without the decorator's caching, we can call the underlying function
    # The decorator might interfere with mocking in some testing scenarios.
    load_model_func = load_model.__wrapped__

    # Configure mocks to return dummy objects
    mock_tokenizer.return_value = "dummy_tokenizer"
    mock_model.return_value = "dummy_model"
    mock_processor.return_value = "dummy_processor"

    model_id = "distilbert-base-uncased"
    processor, tokenizer, model = load_model_func(model_id)

    assert processor == "dummy_processor"
    assert tokenizer == "dummy_tokenizer"
    assert model == "dummy_model"

    mock_processor.assert_called_with(model_id, trust_remote_code=True)
    mock_tokenizer.assert_called_with(model_id, trust_remote_code=True)
    mock_model.assert_called_once()


@patch('app.AutoTokenizer.from_pretrained', side_effect=Exception("Model not found"))
def test_load_model_failure(mock_tokenizer):
    """Test if load_model correctly handles a failure (e.g., model not found)."""
    load_model_func = load_model.__wrapped__

    model_id = "invalid/model-id"
    processor, tokenizer, model = load_model_func(model_id)

    assert processor is None
    assert tokenizer is None
    assert model is None
    st.sidebar.error.assert_called()
    st.error.assert_called_with(f"Could not load the model: {model_id}. Please check the model ID and your internet connection.")
