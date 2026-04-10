# MediRoute AI 🩺

MediRoute AI is an end-to-end Machine Learning web application designed to act as an AI triage assistant.

**Architecture:** Model training is done locally on the host machine. The resulting `model.pkl` is then baked into a Docker image strictly for serving the Streamlit frontend.

## Step 1: Train the Model Locally
You must train the model on your machine to generate the `.pkl` file before starting Docker.

1. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt