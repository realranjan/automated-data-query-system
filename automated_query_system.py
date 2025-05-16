# automated_query_system.py

import os
import csv
import json
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime
from dotenv import load_dotenv
# from langchain_community.chat_models import ChatOllama # Removed or commented out
from langchain_google_genai import ChatGoogleGenerativeAI # New
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import ast

# --- Configuration ---
load_dotenv()

# Generalized configurations for user customization
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "product_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

# Allow users to specify their own CSV file and output directory
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "sample_data.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables. Please set it in your .env file or environment.")
    # exit(1) # Uncomment if the API key is mandatory

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 1. CSV Data Management (NO CHANGES IN THIS SECTION) ---
def connect_to_mongo(uri):
    """Establishes connection to MongoDB."""
    try:
        client = MongoClient(uri)
        client.admin.command('ping')
        print("Successfully connected to MongoDB.")
        return client
    except ConnectionFailure:
        print(f"Failed to connect to MongoDB at {uri}. Ensure MongoDB is running.")
        return None

def preprocess_row(row_dict):
    """
    Preprocesses a row from CSV for MongoDB insertion.
    Handles type conversions and cleaning.
    """
    processed = {}
    for key, value in row_dict.items():
        if pd.isna(value) or value == '':
            processed[key] = None
            continue
        if key in ["Price", "Rating"]:
            try:
                processed[key] = float(value)
            except ValueError:
                print(f"Warning: Could not convert {key}='{value}' to float. Storing as string.")
                processed[key] = str(value)
        elif key in ["ReviewCount", "Stock"]:
            try:
                processed[key] = int(float(value))
            except ValueError:
                print(f"Warning: Could not convert {key}='{value}' to int. Storing as string.")
                processed[key] = str(value)
        elif key == "Discount":
            try:
                if isinstance(value, str) and '%' in value:
                    processed[key] = float(value.replace('%', '')) / 100.0
                else:
                    processed[key] = float(value) / 100.0 if float(value) >=1 else float(value)
            except ValueError:
                print(f"Warning: Could not convert Discount='{value}' to float. Storing as string.")
                processed[key] = str(value)
        elif key == "LaunchDate":
            try:
                processed[key] = datetime.strptime(str(value), "%d-%m-%Y")
            except ValueError:
                try:
                    processed[key] = datetime.fromisoformat(str(value))
                except ValueError:
                    print(f"Warning: Could not parse LaunchDate='{value}'. Storing as string.")
                    processed[key] = str(value)
        else:
            processed[key] = str(value)
    return processed

def load_csv_to_mongo(client, db_name, collection_name, csv_file_path):
    """Loads data from CSV file into a MongoDB collection."""
    if not client:
        return False
    db = client[db_name]
    collection = db[collection_name]
    try:
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file not found at '{csv_file_path}'. Please ensure the file exists and update the CSV_FILE_PATH in your .env file.")
            return False

        if collection.count_documents({}) > 0:
            user_choice = input(f"Collection '{collection_name}' already contains data. \nOptions: (S)kip, (R)eplace, (A)ppend? [S]: ").strip().upper()
            if user_choice == 'R':
                print(f"Replacing data in '{collection_name}'...")
                collection.delete_many({})
            elif user_choice == 'A':
                print(f"Appending data to '{collection_name}'...")
            else:
                print("Skipping data loading.")
                return True

        if collection.count_documents({}) == 0 or user_choice in ['R', 'A']:
            print(f"Loading data from '{csv_file_path}' into '{db_name}.{collection_name}'...")
            df = pd.read_csv(csv_file_path)
            records = []
            for _, row in df.iterrows():
                records.append(preprocess_row(row.to_dict()))
            if records:
                collection.insert_many(records)
                print(f"Successfully loaded {len(records)} documents into '{collection_name}'.")
            else:
                print("No records found in CSV to load.")
        return True
    except Exception as e:
        print(f"An error occurred during CSV loading: {e}")
        return False

def get_collection_schema_description(client, db_name, collection_name):
    """
    Generates a string description of the collection schema for the LLM.
    Tries to infer types from a sample document or CSV headers.
    """
    if not client:
        return "Schema not available (no MongoDB connection)."
    db = client[db_name]
    collection = db[collection_name]
    sample_doc = collection.find_one()

    if not sample_doc:
        try:
            df = pd.read_csv(CSV_FILE_PATH, nrows=1)
            headers = df.columns.tolist()
            schema_parts = []
            for header in headers:
                if header in ["Price", "Rating", "Discount"]: type_hint = "Float"
                elif header in ["ReviewCount", "Stock"]: type_hint = "Integer"
                elif header == "LaunchDate": type_hint = "Date (YYYY-MM-DD format for queries)"
                else: type_hint = "String"
                schema_parts.append(f"- {header}: {type_hint}")
            return "The collection fields are derived from CSV headers:\n" + "\n".join(schema_parts)
        except Exception:
            return "Schema not available (sample document not found and CSV not accessible)."

    schema_parts = []
    for key, value in sample_doc.items():
        if key == "_id": continue
        field_type = type(value).__name__
        if isinstance(value, float): field_type = "Float"
        elif isinstance(value, int): field_type = "Integer"
        elif isinstance(value, str): field_type = "String"
        elif isinstance(value, datetime): field_type = "Date (query values as YYYY-MM-DD)"
        elif isinstance(value, list): field_type = "Array"
        elif isinstance(value, dict): field_type = "Object"
        if key == "Discount": field_type += " (e.g., 0.1 for 10%)"
        if key == "LaunchDate": field_type = "Date (query values as YYYY-MM-DD, e.g., '2022-01-15')"
        schema_parts.append(f"- {key}: {field_type}")
    return "The MongoDB collection has documents with the following fields and inferred types:\n" + "\n".join(schema_parts)


# --- 2. Dynamic Query Generation using LLM (MODIFIED SECTION) ---

def generate_mongodb_query_with_llm(user_question, schema_description):
    """
    Generates a MongoDB query dictionary using Google Gemini LLM.
    Returns a tuple (filter_dict, sort_dict) or (None, None) on error.
    """
    print(f"\nAttempting to generate MongoDB query for: \"{user_question}\" using Gemini ({GEMINI_MODEL})")

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY is not configured. Cannot use Gemini API.")
        return None, None

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )
    except Exception as e:
        print(f"Error initializing Google Gemini LLM: {e}. Check API key and model name.")
        return None, None

    # Define the system prompt content as a raw string.
    # {schema_description} is a LangChain template variable.
    # Literal curly braces for JSON examples are escaped as {{ and }}.
    system_prompt_content = """
You are an expert AI assistant that translates natural language questions into MongoDB query language.
Your goal is to generate Python dictionaries for the 'filter' and 'sort' components of a MongoDB query.

Schema Information:
{schema_description}

Query Generation Guidelines:
1.  Output Format: Respond ONLY with a single JSON object string. This JSON object should have two keys: "filter" and "sort".
    - "filter": A Python dictionary for the MongoDB find() filter.
    - "sort": A Python dictionary for the MongoDB sort() operation (e.g., {{ "Price": -1 }} for descending). Use `null` or an empty dictionary if no sorting is implied.
    Example JSON output: {{ "filter": {{ "Category": "Electronics", "Price": {{ "$lt": 50 }} }}, "sort": {{ "Price": -1 }} }}
    Another example (no sort): {{ "filter": {{ "Brand": "Nike" }}, "sort": null }}

2.  Field Types:
    - String fields (e.g., ProductName, Category, Brand): For "contains" or partial matches, use {{ "$regex": "pattern", "$options": "i" }}. For exact matches, use direct string comparison.
    - Numeric fields (Price, Rating, ReviewCount, Stock, Discount): Use operators like `$eq`, `$gt`, `$gte`, `$lt`, `$lte`. Remember `Discount` is a float (0.1 for 10%).
    - Date fields (LaunchDate): Expect user dates like "January 1, 2022". Convert these to "YYYY-MM-DD" string format for the JSON output. The calling Python code will handle datetime conversion from this string.
      Example: if user says "after January 1, 2022", generate filter like {{ "LaunchDate": {{ "$gt": "2022-01-01" }} }}

3.  Logical Operators:
    - `AND`: Implicit by listing multiple conditions in the filter dict.
    - `OR` for a single field (e.g., brand 'Nike' or 'Sony'): Use `$in` -> {{ "Brand": {{ "$in": ["Nike", "Sony"] }} }}
    - `OR` for different fields: Use `$or` -> {{ "$or": [{{ "Category": "Sports" }}, {{ "Price": {{ "$lt": 30 }} }}] }}

4.  Specific terms:
    - "in stock": Assume this means `Stock > 0`.
    - "discount of X% or more": `Discount >= (X/100)`.

5.  If the question is ambiguous or cannot be translated, output: {{ "filter": {{ "error": "Could not generate query" }}, "sort": null }}

Strictly adhere to providing ONLY the JSON object string as your response. No other text or explanations.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),  # Use the raw string template
        ("human", "{user_question}")
    ])

    chain = prompt_template | llm | StrOutputParser()
    
    try:
        print("Sending request to Gemini LLM...")
        # Provide values for all LangChain template variables used in prompt_template
        response_str = chain.invoke({
            "user_question": user_question,
            "schema_description": schema_description # This is the actual schema string
        })
        print(f"LLM Raw Response: {response_str}")

        if response_str.startswith("```json"):
            response_str = response_str[7:]
            if response_str.endswith("```"):
                response_str = response_str[:-3]
        
        response_str = response_str.strip()

        query_components = json.loads(response_str)
        
        filter_dict = query_components.get("filter")
        sort_dict = query_components.get("sort")

        if filter_dict is None:
            print("Error: LLM response did not contain a 'filter' key or was not valid JSON.")
            return None, None
        if "error" in filter_dict:
            print(f"LLM indicated an error: {filter_dict['error']}")
            return None, None

        def convert_dates_in_filter(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        try:
                            dt_val = datetime.strptime(value, "%Y-%m-%d")
                            obj[key] = dt_val 
                        except ValueError:
                            obj[key] = convert_dates_in_filter(value) 
                    elif isinstance(value, dict) or isinstance(value, list):
                        obj[key] = convert_dates_in_filter(value)
            elif isinstance(obj, list):
                return [convert_dates_in_filter(item) for item in obj]
            return obj

        filter_dict = convert_dates_in_filter(filter_dict)
        
        if not sort_dict:
            sort_dict = None

        return filter_dict, sort_dict

    except json.JSONDecodeError as e:
        print(f"Error decoding LLM JSON response: {e}. Response was: {response_str}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during LLM query generation: {e}")
        if "API key not valid" in str(e): # Example of more specific error check
            print("Please check your GOOGLE_API_KEY.")
        # You might want to re-raise or log the full traceback for debugging complex errors
        # import traceback
        # print(traceback.format_exc())
        return None, None

def log_generated_query(question, filter_query, sort_query):
    """
    Logs the generated query to a file for auditing and debugging purposes.
    """
    try:
        with open(QUERIES_LOG_FILE, 'a') as log_file:
            log_file.write(f"Question: {question}\n")
            log_file.write(f"Filter Query: {json.dumps(filter_query, default=str)}\n")
            log_file.write(f"Sort Query: {json.dumps(sort_query, default=str)}\n")
            log_file.write("\n")
        print("Query logged successfully.")
    except Exception as e:
        print(f"Error logging query: {e}")

# --- 3. Data Retrieval and Presentation (NO CHANGES IN THIS SECTION) ---
def execute_mongo_query(client, db_name, collection_name, query_filter, query_sort=None):
    """Executes the MongoDB query and returns the results."""
    if not client:
        print("Cannot execute query: No MongoDB connection.")
        return None
    if not query_filter:
        print("Cannot execute query: Filter is empty or invalid.")
        return None
    db = client[db_name]
    collection = db[collection_name]
    try:
        print(f"Executing Filter: {query_filter}")
        if query_sort:
            print(f"Applying Sort: {query_sort}")
            # Pymongo sort expects a list of tuples or a dictionary
            if isinstance(query_sort, dict):
                sort_list = list(query_sort.items())
            else: # Assuming it's already in list of tuples format if not dict
                sort_list = query_sort 
            cursor = collection.find(query_filter).sort(sort_list)
        else:
            cursor = collection.find(query_filter)
        results = list(cursor)
        print(f"Found {len(results)} documents.")
        return results
    except OperationFailure as e:
        print(f"MongoDB operation failed: {e.details.get('errmsg', e)}")
        return None
    except Exception as e:
        print(f"An error occurred during query execution: {e}")
        return None

def display_data(data):
    """Presents data in a human-readable format."""
    if not data:
        print("No data to display.")
        return
    df = pd.DataFrame(data)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    print("\nQuery Results:")
    if df.empty:
        print("No matching documents found.")
    else:
        print(df.to_string(index=False))

def save_data_to_csv(data, filename):
    """Saves the retrieved data into a new CSV file."""
    if not data:
        print("No data to save.")
        return
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        df = pd.DataFrame(data)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        df.to_csv(filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Data successfully saved to '{filepath}'")
    except Exception as e:
        print(f"Error saving data to CSV '{filepath}': {e}")

# --- 4. User Interaction and Main Application Logic (Minor change for initial API key check) ---

def main():
    """Main function to run the automated query system."""
    print("Starting Automated Data Query and Retrieval System...")

    if not GOOGLE_API_KEY: # Early exit if key is absolutely required and not found
        print("Fatal Error: GOOGLE_API_KEY is not set. Please configure it in your .env file or environment variables and restart.")
        return

    client = connect_to_mongo(MONGO_URI)
    if not client:
        print("Exiting application due to MongoDB connection failure.")
        return

    if not load_csv_to_mongo(client, DB_NAME, COLLECTION_NAME, CSV_FILE_PATH):
        print("Warning: CSV data loading faced issues. Querying might be affected if collection is empty.")
    
    schema_desc = get_collection_schema_description(client, DB_NAME, COLLECTION_NAME)
    print(f"\nSchema for LLM:\n{schema_desc}\n")

    test_cases = [
        {
            "id": "test_case_1",
            "question": "Find all products with a rating below 4.5 that have more than 200 reviews and are offered by the brand 'Nike' or 'Sony'."
        },
        {
            "id": "test_case_2",
            "question": "Which products in the Electronics category have a rating of 4.5 or higher and are in stock?"
        },
        {
            "id": "test_case_3",
            "question": "List products launched after January 1, 2022, in the Home & Kitchen or Sports categories with a discount of 10% or more, sorted by price in descending order."
        }
    ]

    print("\n--- Running Predefined Test Cases ---")
    for test_case in test_cases:
        print(f"\nProcessing {test_case['id']}: {test_case['question']}")
        filter_q, sort_q = generate_mongodb_query_with_llm(test_case['question'], schema_desc)
        log_generated_query(test_case['question'], filter_q, sort_q)
        if filter_q:
            results = execute_mongo_query(client, DB_NAME, COLLECTION_NAME, filter_q, sort_q)
            if results is not None:
                save_data_to_csv(results, f"{test_case['id']}_results.csv")
            else:
                print(f"Query execution failed or returned no data for {test_case['id']}.")
        else:
            print(f"Could not generate a valid query for {test_case['id']}.")
    print("\n--- Test Case Execution Finished ---")
    
    print("\n--- Interactive Query Mode ---")
    print("Type 'exit' to quit.")
    while True:
        try:
            user_question = input("\nEnter your question about the data: ").strip()
            if user_question.lower() == 'exit':
                break
            if not user_question:
                continue

            filter_q, sort_q = generate_mongodb_query_with_llm(user_question, schema_desc)
            log_generated_query(user_question, filter_q, sort_q)

            if filter_q:
                results = execute_mongo_query(client, DB_NAME, COLLECTION_NAME, filter_q, sort_q)
                if results is not None:
                    if not results:
                        print("No documents found matching your query.")
                    else:
                        while True:
                            action = input("Display results (D), Save to CSV (S), or Neither (N)? [D/S/N]: ").strip().upper()
                            if action == 'D':
                                display_data(results)
                                break
                            elif action == 'S':
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"query_results_{timestamp}.csv"
                                save_data_to_csv(results, filename)
                                break
                            elif action == 'N':
                                break
                            else:
                                print("Invalid choice. Please enter D, S, or N.")
                else:
                    print("Query execution failed.")
            else:
                print("Could not generate a query from your question. Please try rephrasing.")
        
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the interactive loop: {e}")

    if client:
        client.close()
        print("\nMongoDB connection closed.")
    print("Application finished.")

if __name__ == "__main__":
    main()
