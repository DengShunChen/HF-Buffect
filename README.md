# Hugging Face LLM Studio

A simple, local playground built with Streamlit to test and interact with open-source Large Language Models (LLMs) and Multimodal Models from the Hugging Face Hub.

## Features

- **Dynamic Model Loading**: Load any compatible text-generation or multimodal model from Hugging Face by simply providing its Model ID.
- **Hardware Acceleration**: Automatically detects and utilizes NVIDIA (CUDA) or Apple Silicon (MPS) GPUs for faster performance, with a fallback to CPU.
- **Interactive Chat Interface**: A familiar chat-based UI to have conversations with the loaded models.
- **Parameter Control**: Adjust generation parameters like `Temperature` and `Max New Tokens` on the fly.
- **Multimodal Support**: Upload images and ask questions about them using multimodal models like LLaVA.
- **Easy Setup**: Cross-platform startup scripts (`start.sh` for macOS/Linux, `start.bat` for Windows) handle environment setup and dependency installation automatically.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python 3.8+**: Make sure `python3` (or `python`) is in your system's PATH. You can download it from [python.org](https://www.python.org/).
2.  **Git**: Required for cloning the repository. You can get it from [git-scm.com](https://git-scm.com/).
3.  **NVIDIA GPU Users**: A compatible NVIDIA driver is required to use CUDA acceleration.

## How to Run

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Run the Startup Script**

    -   **For macOS and Linux**:
        Open your terminal, make the script executable (only needs to be done once), and run it:
        ```bash
        chmod +x start.sh
        ./start.sh
        ```

    -   **For Windows**:
        Simply double-click the `start.bat` file, or run it from your Command Prompt:
        ```cmd
        start.bat
        ```

3.  **What the Script Does**:
    The first time you run the script, it will:
    - Create a local Python virtual environment in a folder named `venv/`.
    - Activate the environment.
    - Install all the necessary Python packages listed in `requirements.txt`.
    - Finally, it will launch the Streamlit application.

    On subsequent runs, the script will skip the environment creation and package installation steps and launch the app directly.

4.  **Using the Application**:
    - Once launched, your web browser should open to the application's interface.
    - Use the sidebar to enter a Hugging Face Model ID and click "Load Model".
    - Adjust generation parameters as needed.
    - For multimodal models, upload an image before asking your question.
    - Start chatting!

## Default Model

The application will load `google/gemma-2b-it` by default on its first run. You can change this by entering a different model ID in the sidebar.

**Note**: The first time you load a model, it will be downloaded from Hugging Face and cached on your local machine. This may take some time depending on the model size and your internet connection. Subsequent loads of the same model will be much faster.
