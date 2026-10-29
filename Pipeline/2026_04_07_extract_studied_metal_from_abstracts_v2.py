###############################################################################################################
###############################################################################################################

# IMPORTANT: Following evaluation, the two-prompt method
# outperformed this multi-prompt method.
# THE RESULTS OF THIS SCRIPT ARE NOT USED IN THE FINAL CORPUS.

###############################################################################################################
###############################################################################################################



# ======================================================================================
# --- Studied Metal Extraction (Multi-Label Binary Classification) ---
# ======================================================================================

import pandas as pd
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import warnings
import os

import config
import utils

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Binary Prompt Template Definition ---
BINARY_PROMPT_TEMPLATE = """
Analyze the scientific abstract below.
Task: Is {target_description} the MAIN research topic (or one of the main topics) of this study?

CRITICAL RULES:
1. Ignore background comparisons (e.g., if the text says "Unlike Lithium-ion, we propose a Zinc-ion battery", the answer for Lithium is NO).
2. Focus ONLY on the working ion of the battery. Do NOT say YES if the metal is used strictly as an electrode/component material in a different type of battery.
3. Output ONLY "YES" or "NO". Do not provide any explanations.

Abstract:
"{abstract_text}"

Answer:
"""

# --- 2. Targets Dictionary ---
TARGETS = {
    "is_li_ion": "Lithium-ion (Li-ion) batteries",
    "is_na_ion": "Sodium-ion (Na-ion) batteries",
    "is_k_ion": "Potassium-ion (K-ion) batteries",
    "is_zn_ion": "Zinc-ion (Zn-ion) batteries",
    "is_other_metal": "ANY OTHER metal-ion battery (e.g., Magnesium-ion, Aluminum-ion, Calcium-ion, strictly excluding Lithium, Sodium, Potassium, and Zinc)",
    "is_irrelevant": "systems completely UNRELATED to metal-ion batteries (e.g., lead-acid, supercapacitors, fuel cells, or general non-battery topics)"
}

# --- 3. Mapping to Final Tags ---
TAG_MAP = {
    "is_li_ion": "LI-ION",
    "is_na_ion": "NA-ION",
    "is_k_ion": "K-ION",
    "is_zn_ion": "ZN-ION",
    "is_other_metal": "OTHER-METAL",
    "is_irrelevant": "IRRELEVANT"
}

df_data = utils.get_df_data()

async def extrair_metal_async():
    """
    Main asynchronous function to extract binary labels for studied metals.
    """
    print(f"Starting Multi-Label Binary Extraction with model '{config.LLM_ADVANCED_MODEL}'")

    try:
        client_vllm = AsyncOpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_API_KEY)
        print("vLLM client (asynchronous) connected.")
    except Exception as e:
        print(f"ERROR: Could not connect to vLLM client: {e}")
        return

    df_to_process = df_data.copy()
    
    # Create the 6 boolean columns in the DataFrame if they do not exist
    for col in TARGETS.keys():
        if col not in df_data.columns:
            df_data[col] = False

    print(f"Processing {len(df_to_process)} documents (6 prompts per document).")

    CONCURRENCY_LIMIT = 60 
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def ask_vllm(prompt):
        """ Helper function to make the API call with semaphore control """
        async with semaphore:
            try:
                response = await client_vllm.chat.completions.create(
                    model=config.LLM_ADVANCED_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0 # Keep zero to enforce exact YES/NO output
                )
                return response.choices[0].message.content.strip().upper()
            except Exception as e:
                return "ERROR"

    async def process_row(index, row):
        """ Process a single row by firing all prompts simultaneously. """
        abstract_text = row.get(config.COLUMN_ABSTRACT, '')
        
        # If there is no abstract, mark all as false/irrelevant
        if not isinstance(abstract_text, str) or not abstract_text.strip():
            return index, {k: False for k in TARGETS.keys()}

        tasks = []
        keys = list(TARGETS.keys())
        
        for key in keys:
            prompt = BINARY_PROMPT_TEMPLATE.format(
                target_description=TARGETS[key],
                abstract_text=abstract_text
            )
            tasks.append(ask_vllm(prompt))

        results = await asyncio.gather(*tasks)
        
        row_results = {}
        for key, res in zip(keys, results):
            if "YES" in res:
                row_results[key] = True
            else:
                row_results[key] = False

        # Consistency validation for IRRELEVANT
        has_any_metal = any([row_results[k] for k in ["is_li_ion", "is_na_ion", "is_k_ion", "is_zn_ion", "is_other_metal"]])
        if not has_any_metal:
            row_results["is_irrelevant"] = True
        if has_any_metal:
            row_results["is_irrelevant"] = False

        return index, row_results

    tasks = [process_row(index, row) for index, row in df_to_process.iterrows()]
    results_list = []
    
    print("Starting bulk asynchronous processing...")
    for f in async_tqdm.as_completed(tasks, total=len(tasks), desc="Extracting binary classifications"):
        results_list.append(await f)
        
    print("\nProcessing completed. Updating DataFrame in memory...")
    for index, row_results in results_list:
        for col_name, value in row_results.items():
            df_data.loc[index, col_name] = value

    # --- Generate comma-separated aggregated column ---
    print("\nGenerating aggregated 'studied_metal_metal_label' column...")
    
    def construir_label(row):
        # Filter only columns marked as True and get the corresponding Tag
        tags = [tag for col, tag in TAG_MAP.items() if row.get(col) == True]
        
        # Extra safeguard: if no flag was marked (due to extraction error), default to IRRELEVANT
        if not tags:
            tags = ["IRRELEVANT"]
            
        return ", ".join(tags)

    # Apply function to each row of the DataFrame
    df_data['studied_metal_metal_label'] = df_data.apply(construir_label, axis=1)

    print("\nBinary extraction and tag generation completed!")
    
    utils.export_df_to_csv(df_data)
    utils.save_df_data(df_data)

if __name__ == "__main__":
    asyncio.run(extrair_metal_async())
    print("Script 5 finished.")
