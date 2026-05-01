# Named Entity Recognition (NER) using LSTM

## Overview
This project implements a deep learning-based Named Entity Recognition (NER) model utilizing Long Short-Term Memory (LSTM) architecture and the CoNLL2003 dataset. The model accurately identifies and classifies entities such as names (PER), locations (LOC), organizations (ORG), and miscellaneous entities (MISC) in text. The solution is integrated with Streamlit, providing a seamless and interactive user interface for real-time text analysis.

## Features
- **LSTM-Based NER:** Leverages deep learning for improved entity recognition.
- **Pretrained Model:** Uses the `ner_lstm_model.h5` trained on CoNLL2003 dataset.
- **SpaCy Integration:** Leverages spaCy's `en_core_web_sm` model for tokenization and preprocessing.
- **Real-Time Predictions:** Processes and classifies named entities dynamically.
- **Interactive Streamlit UI:** User-friendly interface for text analysis with visual entity highlighting.
- **Continuous Learning:** Option to update the model with new training data.
- **Scalable & Efficient:** Optimized for accuracy and performance on CPU/GPU.

## Entity Labels
- **0:** Outside (O) - Not an entity
- **1:** Beginning-Person (B-PER)
- **2:** Inside-Person (I-PER)
- **3:** Beginning-Organization (B-ORG)
- **4:** Inside-Organization (I-ORG)
- **5:** Beginning-Location (B-LOC)
- **6:** Inside-Location (I-LOC)
- **7:** Beginning-Miscellaneous (B-MISC)
- **8:** Inside-Miscellaneous (I-MISC)

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.10+
- pip

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhisheksoni884/Named-Entity-Recognition.git
   cd Named-Entity-Recognition
   ```

2. **Create and activate a Python virtual environment (optional but recommended):**
   ```bash
   python -m venv ner_env
   source ner_env/bin/activate  # On Windows: ner_env\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the required spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Required Files
Ensure the following files are present in the project directory:
- `app.py` - Main Streamlit application
- `ner_lstm_model.h5` - Pre-trained LSTM model for NER
- `model_data.pkl` - Model metadata (tokenizer, tags, indices)
- `requirements.txt` - Python dependencies

## Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface:**
   - Open your browser and navigate to `http://localhost:8501`
   - You should see the NER application

3. **Analyze Text:**
   - Enter a sentence in the input field
   - Click the "Predict" button
   - View the extracted named entities with color-coded highlighting

4. **Continuous Learning (Optional):**
   - Enter a new sentence and corresponding labels (space-separated integers, e.g., `0 0 0 3 4`)
   - Click "Update Model" to fine-tune the model with new data

## Dependencies

The project requires the following Python packages:
- **TensorFlow 2.18** - Deep learning framework
- **Keras** - Neural network API
- **SpaCy 3.7.2** - NLP library for tokenization
- **Streamlit** - Web framework for building the UI
- **NumPy 1.26.4** - Numerical computations
- **Pandas** - Data manipulation
- **PyArrow** - Data serialization

See `requirements.txt` for complete dependencies.

## Model Information
- **Architecture:** LSTM (Long Short-Term Memory) Neural Network
- **Training Dataset:** CoNLL 2003
- **Max Sequence Length:** 113 tokens
- **Input:** Tokenized text sequences
- **Output:** NER tags for each token

## Troubleshooting

### CUDA/GPU Warnings
If you see warnings about CUDA or GPU not being available, this is normal on CPU-only systems. TensorFlow will automatically use the CPU for inference.

### Missing SpaCy Model
If you get an error about `en_core_web_sm` not found, run:
```bash
python -m spacy download en_core_web_sm
```

### Model File Not Found
Ensure `ner_lstm_model.h5` and `model_data.pkl` are in the project directory.

### Port Already in Use
If port 8501 is already in use, Streamlit will automatically try the next available port (usually 8502, 8503, etc.).

### Long Text/Paragraph Handling
The model was trained with a maximum sequence length of 113 tokens. When you enter text longer than this:
- **Text is truncated** to the first 113 tokens during processing
- **NER predictions** are generated for all available tokens
- **Long paragraphs work fine** - they are automatically capped to the model's capacity

This is by design - the model can only predict up to 113 tokens at a time.

## Performance Notes
- First run may take 10-15 seconds as TensorFlow and SpaCy models are loaded
- Predictions are typically fast (<1 second per sentence) on modern hardware
- For large batch predictions, consider batch processing

## Future Improvements
- Support for multiple languages
- Fine-tuned transformer models (BERT, RoBERTa)
- Batch processing API
- Model export to ONNX format
- Docker containerization

## License
This project is based on the CoNLL 2003 dataset and uses open-source libraries.

## Author
Original implementation by Abhishek Soni

