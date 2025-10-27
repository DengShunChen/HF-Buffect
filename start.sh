#!/bin/bash

# Simple script to set up the environment and run the LLM Studio app.

# --- Configuration ---
PYTHON_CMD="python3" # Or "python" if that's your command for Python 3

# --- Functions ---
function print_info() {
    echo "INFO: $1"
}

function print_error() {
    echo "ERROR: $1" >&2
}

# --- Main Script ---
# 1. Check if Python 3 is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    print_error "Python 3 is not found. Please install Python 3 and try again."
    exit 1
fi

# 2. Check if venv directory exists
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    print_info "Python virtual environment not found. Creating one..."
    if ! $PYTHON_CMD -m venv $VENV_DIR; then
        print_error "Failed to create the virtual environment."
        exit 1
    fi
    print_info "Virtual environment created in '$VENV_DIR/'."
fi

# 3. Activate the virtual environment
print_info "Activating the virtual environment..."
source "$VENV_DIR/bin/activate"

# 4. Install dependencies
print_info "Installing dependencies from requirements.txt..."
if ! pip install -r requirements.txt; then
    print_error "Failed to install dependencies. Please check requirements.txt and your internet connection."
    # Deactivate on failure
    deactivate
    exit 1
fi

# 5. Run the Streamlit application
print_info "All set! Starting the HF LLM Studio..."
print_info "You can close this window to stop the application."
streamlit run app.py

# 6. Deactivate the virtual environment upon closing the app (optional)
print_info "Application closed. Deactivating virtual environment."
deactivate
