# Automated Data Query and Retrieval System (with Google Gemini)

This system allows users to query data loaded from a CSV file into a MongoDB database using natural language. It leverages Google's Gemini API via LangChain, CSV, and MongoDB.

## Features

*   **CSV Data Ingestion**: Loads data from a specified CSV file into a MongoDB collection.
*   **Dynamic Query Generation**: Uses Google Gemini API (e.g., `gemini-pro`) to translate natural language user questions into MongoDB queries.
*   **Data Retrieval**: Executes the generated queries against the MongoDB database.
*   **Data Presentation**:
    *   Displays retrieved data in a user-friendly table format in the console.
    *   Saves retrieved data to a new CSV file.
*   **Test Case Execution**: Automatically runs predefined test cases and saves their results to CSV files.
*   **Query Logging**: Logs user questions and the corresponding LLM-generated MongoDB queries to `Queries_generated.txt`.
*   **Error Handling**: Basic error management for file operations, DB connectivity, and LLM API interactions.

## Prerequisites

1.  **Python 3.8+**: Ensure Python is installed.
2.  **MongoDB Server**: A MongoDB instance must be running and accessible. By default, the script connects to `mongodb://localhost:27017/`.
3.  **Google Cloud Project and Gemini API Key**:
    *   You need a Google Cloud Project with the "Vertex AI API" or "Generative Language API" enabled.
    *   Generate an API key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   **Important**: Keep your API key secure. Do not commit it to version control.

## Setup

1.  **Clone/Download Files**:
    Place `automated_query_system.py`, `sample_data.csv` (or your CSV), and create a `.env` file in the same directory.

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    Ensure `requirements.txt` is updated (see above) and run:
    ```bash
    pip install -r requirements.txt
    ```
    (Key packages: `pymongo`, `pandas`, `langchain`, `langchain-google-genai`, `python-dotenv`)

4.  **Configure Environment**:
    Create a `.env` file in the project root and add your configurations:
    ```env
    MONGO_URI="mongodb://localhost:27017/"
    DB_NAME="product_db"
    COLLECTION_NAME="products"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY" # Replace with your actual API key
    GEMINI_MODEL="gemini-pro" # Or gemini-1.5-pro-latest, gemini-1.0-pro etc.
    CSV_FILE_PATH="sample_data.csv" # Path to your input CSV file
    OUTPUT_DIR="output" # Directory to save output files
    ```

5.  **Sample Data**:
    The system uses `sample_data.csv` by default. You can replace this file with your own CSV file or update the `CSV_FILE_PATH` in the `.env` file to point to your custom file.

6.  **Output Directory**:
    By default, the system saves output files in the `output` directory. You can change this by updating the `OUTPUT_DIR` in the `.env` file.

## Running the System

Execute the main Python script:
```bash
python automated_query_system.py
```