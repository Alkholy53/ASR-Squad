import json
import os
import re
from collections import defaultdict
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from tqdm.auto import tqdm

def write_processed_manifest(data, original_path, output_dir="/kaggle/working/"):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    filepath = os.path.join(output_dir, new_manifest_name)
    write_manifest(filepath, data)
    print(f"Finished writing manifest: {filepath}")
    return filepath

# Calculate the character set
def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

# Remove special characters
def remove_special_characters(data):
    # Arabic specific punctuation and special characters
    chars_to_ignore_regex = r"[^\u0600-\u06FF\s]"  # This will keep only Arabic characters and spaces
    data["text"] = re.sub(chars_to_ignore_regex, " ", data["text"])
    data["text"] = re.sub(r" +", " ", data["text"]).strip()  # Merge multiple spaces
    return data

# Replace diacritics
def replace_diacritics(data):
    data["text"] = re.sub(r"[إأآا]", "ا", data["text"])
    data["text"] = re.sub(r"[يى]", "ي", data["text"])
    data["text"] = re.sub(r"[ؤ]", "و", data["text"])
    data["text"] = re.sub(r"[ة]", "ه", data["text"])
    data["text"] = re.sub(r"[ئ]", "ي", data["text"])
    data["text"] = re.sub(r" +", " ", data["text"]).strip()  # Merge multiple spaces
    return data

# Remove out-of-vocabulary characters
def remove_oov_characters(data):
    oov_regex = r"[^\u0600-\u06FF\s]"  # This will keep only Arabic characters and spaces
    data["text"] = re.sub(oov_regex, "", data["text"]).strip()
    data["text"] = re.sub(r" +", " ", data["text"]).strip()  # Merge multiple spaces
    return data

# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest!")
    return manifest

# List of pre-processing functions
PREPROCESSORS = [
    remove_special_characters,
    replace_diacritics,
    remove_oov_characters,
]

manifest_file = "/kaggle/input/just-json/train.json"
data = read_manifest(manifest_file)

# Apply preprocessing
data_processed = apply_preprocessors(data, PREPROCESSORS)

# Write new manifest
manifest_cleaned = write_processed_manifest(data_processed, manifest_file)
