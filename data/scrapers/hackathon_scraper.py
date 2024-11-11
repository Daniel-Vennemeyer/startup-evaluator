import os
import pandas as pd
import json
import re

# Define the folder containing CSV files
folder_path = 'scrapers/hackathon_data'

# Initialize a list to hold all projects data
all_projects_data = []

# Function to extract the "What it does" section from the project description
def extract_what_it_does(description):
    # Use regex to find the section
    if isinstance(description, str):
        match = re.search(r"(What it does)(.*?)(How I built it|Project scope|How we are building it|Target audience|How it works|How we built it|Technology Stack|What's next|how does it work|How We built it|Challenges we ran into|More on Machine Learning|Challenges I ran into|$)", description, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(2).strip().strip('"')
    return "No description provided."

# Define a function to process each CSV file
def process_csv_file(file_path):
    df = pd.read_csv(file_path)
    
    # Extract relevant fields from each row
    for _, row in df.iterrows():
        project_data = {
            "title": row.get("Project Title", ""),
            "description": extract_what_it_does(row.get("About The Project", "")),  # Extract "What it does" section
        }
        if project_data["description"] and project_data not in all_projects_data and project_data["description"] != "No description provided.":
            all_projects_data.append(project_data)

# Process each file in the 'hackathon_data' folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_name}")
        process_csv_file(file_path)

# Convert the data to JSON and save to a file
output_path = "hackathon_projects.json"
with open(output_path, "w") as f:
    json.dump(all_projects_data, f, indent=4)

print(f"Data extraction complete. Saved to {output_path}")