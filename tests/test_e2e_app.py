import pytest
from playwright.sync_api import Page, expect
import subprocess
import time

# --- Fixtures ---

import socket
from contextlib import closing

def is_port_in_use(port: int) -> bool:
    """Checks if a local port is in use."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        return s.connect_ex(('localhost', port)) == 0

@pytest.fixture(scope="session")
def streamlit_app():
    """
    Fixture to start and stop the Streamlit app, waiting for it to be ready.
    """
    command = ["streamlit", "run", "app.py", "--server.port", "8501"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for the Streamlit server to be ready
    port = 8501
    timeout = 60  # seconds
    start_time = time.time()

    print(f"Waiting for Streamlit app to be available on port {port}...")

    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            print(f"✅ Streamlit app is up and running on port {port}!")
            yield process
            process.terminate()
            process.wait()
            return
        time.sleep(1) # Check every second

    # If the loop finishes, the app failed to start
    stdout, stderr = process.communicate()
    process.terminate()
    process.wait()
    pytest.fail(
        f"Streamlit app failed to start within {timeout} seconds.\n"
        f"STDOUT:\n{stdout}\n"
        f"STDERR:\n{stderr}",
        pytrace=False
    )

# --- E2E Tests ---

def test_app_full_flow_with_tiny_model(streamlit_app, page: Page):
    """
    Tests the full user flow with a real, tiny, and valid model.
    1. Navigates to the app.
    2. Enters a tiny model ID in the sidebar.
    3. Clicks "Load Model".
    4. Waits for the model to load successfully.
    5. Enters a prompt in the chat input and submits.
    6. Verifies that the assistant displays a non-empty response.
    """
    app_url = "http://localhost:8501"
    # Using a purpose-built tiny model with a chat template.
    tiny_model_id = "HuggingFaceH4/tiny-random-LlamaForCausalLM"

    # 1. Navigate to the app
    page.goto(app_url)
    expect(page.locator("h1")).to_have_text("🧠 Hugging Face LLM Studio", timeout=30000)

    # 2. Enter the tiny model ID
    model_input = page.locator('input[aria-label="Enter Hugging Face Model ID"]')
    model_input.fill(tiny_model_id)

    # 3. Click "Load Model"
    page.get_by_role("button", name="Load Model").click()

    # 4. Wait for the model to load successfully.
    sidebar = page.locator('[data-testid="stSidebar"]')
    success_message = sidebar.locator(
        f'[data-testid="stAlert"]:has-text("✅ Model {tiny_model_id} loaded successfully!")'
    )
    expect(success_message).to_be_visible(timeout=180000)

    # 5. Enter a prompt and submit
    chat_input = page.locator('textarea[data-testid="stChatInputTextArea"]')
    expect(chat_input).to_be_visible(timeout=10000)
    prompt = "Hello, what can you do?"
    chat_input.fill(prompt)
    chat_input.press("Enter")

    # 6. Verify assistant response
    # Use a structural locator based on the assistant's avatar, which is more robust than text.
    assistant_response_locator = page.locator('[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])').last
    expect(assistant_response_locator).to_be_visible(timeout=90000) # Increased timeout for first generation

    response_text_locator = assistant_response_locator.locator('div[data-testid="stMarkdownContainer"] p')
    expect(response_text_locator).not_to_be_empty(timeout=10000)

    final_response = response_text_locator.inner_text()
    assert final_response is not None and final_response.strip() != ""
