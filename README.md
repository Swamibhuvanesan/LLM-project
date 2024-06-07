# Custom Knowledge Base Chatbot

This project is a custom knowledge base chatbot built using Streamlit and various NLP models. It supports both question-answering and generative responses based on the provided documents.

## Features

- **Question Answering (QA) Mode**: Provides answers to specific questions using context from loaded documents.
- **Generative Mode**: Generates detailed and accurate responses to questions based on provided documents.
- **Document Loading**: Supports loading documents from URLs and processes them for efficient querying.
- **Caching**: Implements caching for repeated queries to improve performance.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Swamibhuvanesan/LLM-project.git
    cd LLM-project
    ```

2. Create and activate a virtual environment (recommended):

    ```bash
    python -m venv newenv
    source newenv/bin/activate  # On Windows, use `newenv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app-new.py
    ```

2. Open your browser and go to `http://localhost:8501`.

3. Enter document URLs (comma-separated) in the input field to load documents.

4. Choose the mode (QA or Generative).

5. Ask questions and receive answers based on the loaded documents.

## Project Structure

- `app-new.py`: Main application file for Streamlit.
- `utils.py`: Contains utility functions for model loading, document processing, and answering questions.
- `requirements.txt`: List of required Python packages.

## Models Used

- **Sentence Transformer**: For embedding documents and queries.
- **Question Answering Model**: `distilbert-base-uncased-distilled-squad`.
- **Generative Model**: `gpt2`.

## Notes

- **Local Disk Space**: Ensure you have sufficient local disk space to store the downloaded models.
- **Model Installation**: The models will be downloaded and installed locally when the app is first run. This requires an internet connection.
- **Swappable Models**: The models used in this project can be easily swapped for other models from the Hugging Face library, making it flexible for different use cases.
- **License**: The models used are under the Apache 2.0 license, which allows for broad usage with proper attribution.

## Screenshot

<div align="center">
  <img src="https://github.com/Swamibhuvanesan/Other-works/blob/main/resource/QA.png" width="1000" height="500" alt="png">
</div>

<div align="center">
  <img src="https://github.com/Swamibhuvanesan/Other-works/blob/main/resource/Generative.png" width="1000" height="500" alt="png">
</div>

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Thanks to the developers of Streamlit, Hugging Face Transformers, and other open-source libraries used in this project.
