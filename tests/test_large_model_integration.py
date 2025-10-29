import pytest
import os
from app import load_model
import torch

# This test requires a Hugging Face token with access to Llama 3.
# It should be set as an environment variable: HF_TOKEN
# We will skip this test if the token is not available.
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")
REASON_NO_TOKEN = "Hugging Face token not set. Skipping Llama 3 integration test."

# Define the model ID for the test
LLAMA3_MODEL_ID = "meta-llama/Meta-Llama-3-8B"

@pytest.mark.slow
@pytest.mark.skipif(not HUGGING_FACE_TOKEN, reason=REASON_NO_TOKEN)
def test_load_llama3_model():
    """
    Integration test to verify the download and loading of the Llama 3 8B model.
    This is a long-running test and requires significant resources.
    """
    # The load_model function is decorated with @st.cache_resource.
    # We call it directly to test its real behavior, including caching.
    try:
        processor, tokenizer, model = load_model(LLAMA3_MODEL_ID)

        # 1. Check if the components are loaded
        assert tokenizer is not None, "Tokenizer should not be None."
        assert model is not None, "Model should not be None."
        # Llama 3 is a text-only model, so the processor should be None.
        assert processor is None, "Processor should be None for Llama 3."

        # 2. Check the types of the loaded components
        # We can't be too specific due to the transformers library's internal classes,
        # but we can check for base classes.
        from transformers import PreTrainedTokenizer, PreTrainedModel
        assert isinstance(tokenizer, PreTrainedTokenizer), f"Tokenizer is of type {type(tokenizer)}, not PreTrainedTokenizer."
        assert isinstance(model, PreTrainedModel), f"Model is of type {type(model)}, not PreTrainedModel."

        # 3. Check if the model is on the correct device
        # Note: This check assumes the environment for the test has a GPU if available.
        # It dynamically checks the expected device.
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        # For a quantized model with device_map, the device attribute might not be straightforward.
        # Instead, we can check the device of one of its parameters.
        model_device = next(model.parameters()).device
        assert model_device.type == expected_device, f"Model is on device {model_device.type}, but expected {expected_device}."

        # 4. Perform a simple inference test
        prompt = "Hello, world!"
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

        # Generate a response
        outputs = model.generate(**inputs, max_new_tokens=10)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert isinstance(response_text, str), "The decoded response should be a string."
        assert len(response_text) > len(prompt), "The response should be longer than the prompt."

    finally:
        # Clear the cache to ensure subsequent test runs are clean.
        # This is important if tests are run in the same session.
        load_model.clear()
