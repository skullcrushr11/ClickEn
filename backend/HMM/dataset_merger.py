import json
import os

# Directory containing JSON files
json_dir = "sandbox"  # Update with actual path
cheating_database = {}
normal_database = {}

def merge_data(target_database, user_id, user_data):
    if user_id not in target_database:
        target_database[user_id] = {"keyboard_data": [], "mouse_data": []}
    
    target_database[user_id]["keyboard_data"].extend(user_data.get("keyboard_data", []))

def process_files(file_list, target_database):
    for filename in file_list:
        file_path = os.path.join(json_dir, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for user_id, user_data in data.items():
                merge_data(target_database, user_id, user_data)

# Get list of cheating and normal files
cheating_files = [f for f in os.listdir(json_dir) if f.startswith("cheating") and f.endswith(".json")]
normal_files = [f for f in os.listdir(json_dir) if f.startswith("normal") and f.endswith(".json")]

# Process cheating and normal files
process_files(cheating_files, cheating_database)
process_files(normal_files, normal_database)

# Save the merged cheating data
cheating_output_path = os.path.join(json_dir, "cheating.json")
with open(cheating_output_path, 'w') as f:
    json.dump(cheating_database, f, indent=4)

# Save the merged normal data
normal_output_path = os.path.join(json_dir, "normal.json")
with open(normal_output_path, 'w') as f:
    json.dump(normal_database, f, indent=4)

print(f"Cheating data saved to {cheating_output_path}")
print(f"Normal data saved to {normal_output_path}")
