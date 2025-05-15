import os
import random
import gc
from tqdm import tqdm
import pdfplumber

pdf_dir = "./data/climate_text_dataset"
new_train_dir = "./train"
new_test_dir = "./test"
split_ratio = 0.9
random_seed = 42
pages_per_chunk = 10
random.seed(random_seed)

def get_processed_files(output_dir):
    return set(f.replace('.txt', '') for f in os.listdir(output_dir) if f.endswith('.txt'))

def extract_text(pdf_path, dst_path, pages_per_chunk=10):
    try:
        with pdfplumber.open(pdf_path) as pdf, open(dst_path, 'w', encoding='utf-8') as outF:
            num_pages = len(pdf.pages)
        for start_idx in tqdm(range(0, num_pages, pages_per_chunk), desc="Reading pages", leave=False):
            print(f"Processing pages: {start_idx} to {start_idx + pages_per_chunk}")
            with pdfplumber.open(pdf_path) as pdf, open(dst_path, 'w', encoding='utf-8') as outF:
                end_idx = start_idx + pages_per_chunk
                chunk_pages = pdf.pages[start_idx:end_idx]

                # Combine text in this chunk
                chunk_text = []
                for page in chunk_pages:
                    page_text = page.extract_text()
                    if page_text:
                        chunk_text.append(page_text)
                print(f"Processed chunks: {len(chunk_text)}")
                # Write chunk to the file
                if chunk_text:
                    outF.write(' '.join(chunk_text))
                    outF.write('\n')
            del chunk_text
            gc.collect()
        return True
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return False
    finally:
        gc.collect()

def copy_or_process_file(file_name, dst_dir, pdf_dir):
    dst_path = os.path.join(dst_dir, f"{file_name}.txt")
    if os.path.exists(dst_path):
        print(f"Skipping (already processed): {file_name}")
        return

    pdf_path = os.path.join(pdf_dir, file_name)
    print(f"Processing: {file_name}")
    success = extract_text(pdf_path, dst_path, pages_per_chunk=pages_per_chunk)
    if success:
        print(f"Processed: {file_name}")

def process_split(file_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in tqdm(file_list, desc=f"Processing into {output_dir}"):
        copy_or_process_file(file_name, output_dir, pdf_dir)
        gc.collect()

def set_seed(seed):
    random.seed(seed)

if __name__ == "__main__":
    set_seed(random_seed)
    all_pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    random.shuffle(all_pdfs)

    split_idx = int(split_ratio * len(all_pdfs))
    train_files = all_pdfs[:split_idx]
    test_files = all_pdfs[split_idx:]

    process_split(train_files, new_train_dir)
    process_split(test_files, new_test_dir)
