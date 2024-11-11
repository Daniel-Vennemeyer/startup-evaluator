import pandas as pd
import json

def main(negative_example_paths=["hackathon_projects.json"], positive_example_paths=["yc_companies.json"], save_path="data/projects_labeled.csv"):
    # Load the JSON data for hackathon projects

    formatted_data = []
    for file in negative_example_paths:
        with open(file, "r") as f:  # Replace 'projects.json' with your JSON file path
            data = json.load(f)

        for item in data:
            if item.get('name', ''):
                text = f"{item.get('name', '')} - {item.get('description', '')}"
            else: 
                text = f"{item.get('title', '')} - {item.get('description', '')}"
            formatted_data.append({"text": text, "label": 0})

    for file in positive_example_paths:
        # Load the JSON data for YC projects
        with open(file, "r") as f:  # Replace 'projects.json' with your JSON file path
            data = json.load(f)

        for item in data:
            if item:
                # Remove the company name from the start of the description
                description = item.get('description', '')
                name = item.get('name', '')

                # Remove the company name and the word "is" if it directly follows
                if description.startswith(f"{name} is "):
                    # Remove the company name and "is"
                    description = description[len(name) + 3:].strip()
                elif description.startswith(name):
                    # Only remove the company name
                    description = description[len(name):].strip()

                # Capitalize the first letter if it's lowercase
                description = description[:1].upper() + description[1:] if description else description

                # Construct the text field with the modified description
                if item.get('name', ''):
                    text = f"{item.get('name', '')} - {item.get('description', '')}"
                else: 
                    text = f"{item.get('title', '')} - {item.get('description', '')}"
                formatted_data.append({"text": text, "label": 1})

    # Convert to DataFrame
    df = pd.DataFrame(formatted_data)

    # Save to CSV
    df.to_csv(save_path, index=False)

    print("Conversion complete! Saved to 'projects_labeled.csv'")

if __name__ == "__main__":
    main(save_path="data/augmented_labeled.csv", negative_example_paths=["data/raw_data/hackathon_augmented.json", "data/raw_synthetic_data/large_synthetic_negative.json"], positive_example_paths=["data/raw_data/yc_augmented.json", 'data/raw_synthetic_data/large_synthetic_positive.json'])