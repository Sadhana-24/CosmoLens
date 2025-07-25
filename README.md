# CosmoLens

CosmoLens is a multimodal Retrieval-Augmented Generation (RAG) project designed for exploring and understanding astronomical data. It leverages state-of-the-art machine learning models to provide a seamless and intuitive interface for querying vast datasets of celestial images and information.

## Project Overview

The core of CosmoLens is a sophisticated RAG pipeline that combines the power of large language models (LLMs) with the precision of vector-based retrieval. This allows users to ask questions in natural language, and even use images as part of their queries, to receive accurate and contextually relevant answers from a massive corpus of astronomical data.

## System Architecture

The CosmoLens system is divided into two main components: an offline data preparation and indexing pipeline, and an online query processing pipeline.

![CosmoLens System Architecture](docs/GenAI_Architecture_Diagram.jpeg)

### Offline Data Preparation and Indexing

1.  **Source Datasets:** We utilize a diverse range of astronomical datasets, including images and metadata from the European Space Agency (ESA)/Hubble Space Telescope, the Astronomy Picture of the Day (APOD), and the European Southern Observatory (ESO).

2.  **Image and Text Processing:**
    *   **Image Processing:** Images from our source datasets are processed using the Jina CLIP v2 model to generate 1024-dimensional vector embeddings.
    *   **Text Processing:** Corresponding textual information is processed with Jina Embeddings v3 to create 1024-dimensional vector embeddings.

3.  **Pinecone Vector Database:** The generated image and text embeddings are stored and indexed in a Pinecone vector database. This allows for efficient similarity searches and retrieval of relevant data during the query process. We maintain separate namespaces for different embedding types (`clip-embeddings`, `astrollava-embeddings`, `hubble-embeddings`).

4.  **ID-URL Mapping:** A JSONL file is used to maintain a mapping between the unique identifiers of our data points and their corresponding source URLs. This enables us to provide users with direct links to the original data sources.

### Online Query Processing Pipeline

1.  **User Input:** Users can interact with the system through a user-friendly interface that accepts both text queries and image uploads.

2.  **Image Analysis and Query Formulation:**
    *   **Image Analysis:** If an image is provided as input, it is analyzed by the Gemini Flash model to extract relevant features and information.
    *   **Query Formulation:** The user's input (text and/or image) is formulated into a comprehensive query.

3.  **Query Embedding:** The formulated query is then embedded using the Jina v3 model to generate a vector representation.

4.  **Vector Similarity Search:** The query vector is used to perform a similarity search against the embeddings stored in the Pinecone vector database. The top 10 most similar vectors are retrieved from each relevant namespace.

5.  **Context Aggregation and Generative Synthesis:**
    *   **Context Aggregation:** The retrieved data from the vector search is aggregated to form a rich context for the final response.
    *   **Generative Synthesis:** The aggregated context is then passed to the Gemini Flash LLM, which generates a coherent and informative answer in natural language.

6.  **Response:** The final response, which includes the generated text and the source URLs of the retrieved data, is presented to the user.

### Web Server

The entire online query processing pipeline is served through a FastAPI web server, which uses Pydantic models for data validation and asynchronous processing to handle requests efficiently.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Node.js and npm
*   A Pinecone account and API key
*   A Google AI API key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/CosmoLens.git
    cd CosmoLens
    ```

2.  **Set up the Python environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up the frontend:**
    ```bash
    cd frontend/stargazer-answer-bot
    npm install
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root of the project and add your API keys:
    ```
    PINECONE_API_KEY="your-pinecone-api-key"
    GOOGLE_API_KEY="your-google-api-key"
    ```

### Running the Application

1.  **Start the backend server:**
    ```bash
    uvicorn src.rag_fast_api:app --reload
    ```

2.  **Start the frontend development server:**
    ```bash
    cd frontend/stargazer-answer-bot
    npm run dev
    ```

The application should now be running at `http://localhost:5173`.

## Contributors

This project was developed by:

*   Samarth P
*   Sadhana Shashidhar
*   Sakshi Rajani
*   Sakshi Masand
