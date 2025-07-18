---
title: ChiWell
emoji: 🍃
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# ChiWell🌿: An AI-Powered TCM Wellness Platform

## About The Project

ChiWell is a comprehensive, full-stack web application designed to modernize and democratize access to Traditional Chinese Medicine (TCM) wellness practices. By integrating advanced AI models with a user-friendly interface, the platform provides an interactive and educational ecosystem for users to learn, practice, and explore the principles of TCM.

The project addresses the challenge of making ancient wellness practices accessible and engaging for a modern audience. It moves beyond static content, offering personalized, data-driven feedback and guidance through a suite of powerful, integrated modules.

## Key Features

The platform is built around four core, AI-powered modules:

1.  **Real-Time Ba Duan Jin Practice:** An interactive coaching module that uses a user's webcam and client-side pose estimation (MediaPipe) to provide immediate, quantitative feedback on their Ba Duan Jin exercise form by comparing it to an expert reference in real-time.
2.  **Multimodal Video Analysis:** A sophisticated offline analysis tool where users can upload a video of their practice. Google's Gemini 2.5 Pro model performs a deep, comparative analysis against a standard video to generate a detailed, bilingual coaching report focused on biomechanics and posture, rather than simple action classification.
3.  **Herbal Medicine Recognition:** An image recognition tool powered by a fine-tuned ConvNeXt-Tiny model. Users can upload a photo of a Chinese herb, and the AI will identify it from a comprehensive database.
4.  **Interactive Acupoint Atlas:** A searchable, visual encyclopedia of TCM acupoints. It features a hierarchical navigation system based on meridians and provides detailed information, including anatomical images and descriptions for each point.
5.  **Conversational AI Assistant:** A site-wide, persistent chatbot powered by Gemini 2.5 Flash and LangChain. It acts as a knowledgeable health assistant, answering user questions about TCM, exercises, and general wellness with conversational memory.

## Getting Started

Follow these instructions to set up and run the project on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8 or higher
- `pip` for package management
- An environment with access to the internet to download models and contact Google's APIs.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    A `requirements.txt` file is recommended for a smooth setup. If you don't have one, you can create it from the necessary packages.
    ```bash
    pip install flask torch torchvision Pillow timm google-generativeai langchain-google-genai langchain-core
    ```

4.  **Set up the necessary data and model files:**
    Ensure the following files and directories are correctly placed in your project's root directory:
    - `medicine_model.pth`: The pre-trained weights for the herb recognition model.
    - `class_names.json`: The class labels corresponding to the herb model.
    - `standard_pose_data.json`: The reference keypoints for the real-time practice module.
    - `acupoints_data/`: The directory containing all meridian and acupoint subfolders and data.
    - `static/videos/baduanjin.mp4`: The standard reference video for the multimodal analysis.
    - `templates/`: Directory containing all HTML template files.

## Configuration

The application requires an API key for Google's Gemini models.

1.  **Obtain a Gemini API Key:**
    Visit the [Google AI Studio](https://aistudio.google.com/) to create your API key.

2.  **Set the Environment Variable:**
    You need to set the `GEMINI_API_KEY` environment variable. You can do this in your terminal before running the app:
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    On Windows, use:
    ```bash
    set GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    For a more permanent solution, consider using a `.env` file and a library like `python-dotenv`.

## Usage

Once all the prerequisites, files, and configurations are in place, you can run the Flask application.

From your project's root directory, execute the following command:

```bash
python app.py