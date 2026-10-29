import utils
import config

import pandas as pd
import asyncio
from openai import AsyncOpenAI  # Using the official library
import json
from tqdm.asyncio import tqdm
import time

tqdm.pandas() 

# === 1. Load df_data (main dataset) ===
df_data = utils.get_df_data()


# --- 2. VLLM Constants ---

# VLLM base URL (OpenAI-compatible API)
VLLM_BASE_URL = "http://localhost:8000/v1"
# The 'openai' library requires a key, even if VLLM does not.
VLLM_API_KEY = "vllm" 
MODEL_NAME = "RedHatAI/gemma-3-4b-it-quantized.w8a8"

# Concurrent request limit for VLLM
# Adjust according to your GPU/server capacity
MAX_CONCURRENCY = 100 

# Prompt in English (as defined previously)
PROMPT_TEMPLATE = '''<start_of_turn>user
You are an expert scientific research classifier. Your task is to classify the provided abstract into one of the following two research types.

Here are the category definitions:

1.  **"Computational and simulation studies"**: "use theoretical models and computational tools to predict and analyze the properties and mechanical characteristics without physical experiments"
2.  **"Experimental studies"**: "involve the practical investigation and characterization through laboratory experiments to determine their physical, chemical, and mechanical properties using various analytical techniques"

If the abstract describes both methods, classify it based on the primary contribution of the work. For example, if computational work is only used to support a mainly experimental finding, classify it as 'Experimental studies'.

Analyze the abstract below and respond **only** with a JSON object. The JSON must have a single key named "research_type", and the value must be **exactly** one of the two categories defined above.

**Abstract:**
"""
{abstract_text}
"""

**Classification (JSON only):**
```json
<end_of_turn>
<start_of_turn>model
'''

# --- 3. Asynchronous OpenAI Client ---

# Instantiate the client ONCE. It manages the connection pool.
client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

# --- 4. Asynchronous Classification Function with Semaphore ---

async def fetch_classification(
    semaphore: asyncio.Semaphore, 
    abstract: str
) -> tuple[str, str]:
    """
    Sends an abstract to VLLM using the AsyncOpenAI client,
    controlled by a semaphore.
    Returns (abstract, classification) to ensure accurate mapping.
    """
    if not abstract or not isinstance(abstract, str) or len(abstract.strip()) < 20:
         return (abstract, "Not_Applicable")

    full_prompt = PROMPT_TEMPLATE.format(abstract_text=abstract)
    
    # Acquire semaphore before making the call
    # This ensures that at most MAX_CONCURRENCY calls
    # are active at the same time.
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
                max_tokens=100
            )
            
            llm_output_text = response.choices[0].message.content
            
            # Clean and parse JSON
            json_str = llm_output_text.strip().replace("```json", "").replace("```", "").strip()
            classification_json = json.loads(json_str)
            classification = classification_json.get("research_type", "LLM_Parse_Error")
            
            # Return tuple for mapping
            return (abstract, classification)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing LLM response: {e} | Response: {llm_output_text}")
            return (abstract, "LLM_Parse_Error")
        except Exception as e:
            # Catch API errors (e.g., connection, timeout, etc.)
            print(f"VLLM API Error: {e} | Abstract: {abstract[:50]}...")
            return (abstract, "LLM_Request_Error")

# --- 5. Main Orchestration Function ---

# def classify_without_abstract(row):
    
#     text = " ".join([
#         str(row.get("Document Title", "")),
#         str(row.get("Author Keywords", "")),
#         str(row.get("Keywords Plus®", "")),
#         str(row.get("topic_cluster_1", ""))
#     ]).lower()
    
#     # Computational
#     if any(k in text for k in [
#         "model", "simulation", "dft", "numerical", "theoretical", "machine learning"
#     ]):
#         return "Computational and simulation studies"
    
#     # Experimental
#     if any(k in text for k in [
#         "synthesis", "fabrication", "experimental", "electrochemical",
#         "characterization", "performance", "cycling", "xrd", "sem", "tem"
#     ]):
#         return "Experimental studies"
    
#     return ""

async def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the asynchronous classification of the entire DataFrame.
    """
    start_time = time.time()
    
    # 1. Initialize the new column
    df['research_type'] = None

    # 2. Rule 1: Fill in "Review" rows
    review_mask = df[config.COLUMN_DOCUMENT_TYPE] == "Review"
    df.loc[review_mask, 'research_type'] = "Review"
    print(f"Step 1: {review_mask.sum()} rows marked as 'Review'.")

    # 3. Identify unique abstracts that require classification
    to_classify_mask = df['research_type'].isnull()
    valid_abstracts_mask = df[config.COLUMN_ABSTRACT].notna() & (df[config.COLUMN_ABSTRACT].str.len() > 20)
    final_mask = to_classify_mask & valid_abstracts_mask
    
    unique_abstracts = df.loc[final_mask, config.COLUMN_ABSTRACT].unique()
    
    if len(unique_abstracts) == 0:
        print("No valid abstracts to classify with LLM.")
        return df

    print(f"Step 2: {len(unique_abstracts)} unique abstracts to send to VLLM (from {final_mask.sum()} rows).")

    # 4. Create semaphore and tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = []
    
    for abstract in unique_abstracts:
        tasks.append(fetch_classification(semaphore, abstract))
    
    print(f"Step 3: Sending {len(tasks)} requests to VLLM (max concurrency: {MAX_CONCURRENCY})...")
    
    # 5. Execute tasks with progress bar
    results = await tqdm.gather(*tasks, desc="Classifying Abstracts")

    # 6. Create mapping dictionary
    # 'results' is a list of tuples: [(abs1, type1), (abs2, type2), ...]
    abstract_to_type_map = dict(results)
    
    errors = {k: v for k, v in abstract_to_type_map.items() if "Error" in v or "Not_Applicable" in v}
    if errors:
        print(f"\nWarning: {len(errors)} abstracts failed classification.")

    # 7. Map results back to DataFrame (ensuring accurate association)
    df['research_type'] = df['research_type'].fillna(
        df[config.COLUMN_ABSTRACT].map(abstract_to_type_map)
    )

    # # 8. Perform additional check for rows without abstract or with invalid abstract
    # mask = df['research_type'].isnull()
    # df.loc[mask, 'research_type'] = df.loc[mask].apply(classify_without_abstract, axis=1)

    end_time = time.time()
    print(f"\nStep 4: Mapping completed. Total time: {end_time - start_time:.2f} seconds.")
    return df

# --- 6. Execution ---

if __name__ == "__main__":
    print("Starting asynchronous classification with AsyncOpenAI and Semaphore...")
    
    # Main block to run asyncio
    # Note: In a Jupyter Notebook, you may need to use 'await'
    # instead of 'asyncio.run()' if an event loop is already running.
    df_classified = asyncio.run(classify_dataframe(df_data))
    
    print("\n--- Summary of Research Types ---")
    print(df_classified['research_type'].value_counts())


    print("Saving updated DataFrame...")
    utils.save_df_data(df_classified)
    utils.export_df_to_csv(df_classified)

print("Process completed.")
