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
    git clone <repository_url>
    cd <repository_name>
    ```

2. Install the required dependencies:

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

## Screenshot

![Project Screenshot](path_to_your_image.png)

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the Apache 2.0.

## Acknowledgements

- Thanks to the developers of Streamlit, Hugging Face Transformers, and other open-source libraries used in this project.
