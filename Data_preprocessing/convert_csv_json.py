import csv
import json
import os
import wave
# Define input and output file paths
input_csv =  'train.csv'
audio_folder =  r'adapt'
output_json =  'train.json'
# Function to get the duration of a WAV file
def  get_wav_duration(wav_path):
try:
with wave.open(wav_path, 'r') as wav_file:
frames = wav_file.getnframes()
rate = wav_file.getframerate()
duration = frames /  float(rate)
return duration
except  Exception  as e:
print(f"Error reading {wav_path}: {e}")
return  None

# List all files in the audio folder
all_files = os.listdir(audio_folder)
print(f"Files in '{audio_folder}': {all_files}")

# Initialize a list to hold the formatted data
formatted_data = []
print("Starting to process the CSV file...")

# Read the CSV file and process each row
with  open(input_csv, mode='r', encoding='utf-8') as csvfile:
csvreader = csv.DictReader(csvfile)
row_count =  sum(1  for row in csvreader) # Get the total number of rows
csvfile.seek(0) # Reset the reader to the beginning of the file
next(csvreader) # Skip the header row
for i, row in  enumerate(csvreader, start=1):

# Get the audio filename and transcript from the CSV
audio_filename = row['audio'].strip() # Remove any leading/trailing whitespace
transcript = row['transcript']

# Append .WAV extension if missing
if  not audio_filename.lower().endswith('.wav'):
audio_filename +=  '.wav'

# Construct the full path to the audio file
audio_filepath = os.path.join(audio_folder, audio_filename)

# Debugging: Print the full path of the audio file
print(f"Processing file {audio_filepath}")

# Check if the file exists
if  not os.path.isfile(audio_filepath):
print(f"File not found: {audio_filepath}")
continue

# Get the duration of the audio file
duration = get_wav_duration(audio_filepath)
if duration is  None:
print(f"Skipping file {audio_filepath} due to error in reading duration.")
continue

# Create a dictionary in the desired format
formatted_entry = {
"audio_filepath": audio_filepath,
"duration": duration,
"text": transcript
}

# Add the formatted entry to the list
formatted_data.append(formatted_entry)
print(f"Processed {i}/{row_count} rows")

# Write the formatted data to a JSON file
try:
with  open(output_json, mode='w', encoding='utf-8') as jsonfile:
json.dump(formatted_data, jsonfile, indent=4, ensure_ascii=False)
print(f"Formatted data has been saved to {output_json}")
except  Exception  as e:
print(f"Error writing to {output_json}: {e}")
print("Processing complete.")