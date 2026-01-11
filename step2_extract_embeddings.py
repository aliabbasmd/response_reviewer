import numpy as np
import pandas as pd
import os
import torch
import glob
from transformers import AutoTokenizer, AutoModel

# Define the models
llms = {
    "t5": "t5-small",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base"
    # Llama-2 and Gatortron are omitted here for speed/auth reasons, 
    # but can be added back if you have the hardware/tokens.
}

def extract_embeddings(text_list, model_name):
    """Batch processes text to get embeddings."""
    model_path = llms[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # Use a pipeline approach or batching for efficiency
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        if "t5" in model_name:
            outputs = model.encoder(**inputs)
        else:
            outputs = model(**inputs)
        
        # Mean pooling to get one vector per sentence
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings

output_dir = "generated_data"
csv_files = glob.glob(f"{output_dir}/data_*.csv")

if __name__ == "__main__":
    print(f"Found {len(csv_files)} files. Starting embedding extraction...")
    
    # Process only the first few files as a test (embeddings take time!)
    for filename in csv_files[:5]: 
        df = pd.read_csv(filename)
        eq_type = filename.split('_')[-1].replace('.csv', '')
        
        # Create a text string from categories
        text_data = df.apply(lambda row: f"{row['cat1']} {row['cat2']}", axis=1).tolist()
        
        for llm_name in llms.keys():
            print(f"Processing {filename} with {llm_name}...")
            
            # Extracting in one batch for the file (500 rows)
            embeddings = extract_embeddings(text_data, llm_name)

            # Save as .npy (Binary format for numpy)
            clean_name = os.path.basename(filename).replace('.csv', '')
            save_path = f"{output_dir}/embeddings_{llm_name}_{clean_name}.npy"
            np.save(save_path, embeddings)
            
    print("Embedding extraction complete for sample files.")
