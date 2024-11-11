from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import google.generativeai as genai
import os
from dotenv import load_dotenv
# Import the main function from model.py to train the model if needed
from model import main as train_model

# Load environment variables from .env file (optional)
load_dotenv()

app = Flask(__name__)

# Define the model path
model_path = "./augmented_fine_tuned_roberta_model"

# Check if model exists, and if not, train the model
if not os.path.exists(model_path):
    print("Model not found. Running model training script...")
    train_model()  # Fine-tunes the model

# Load the fine-tuned model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Load the pre-ranked projects
ranking_path = "ranked_projects.csv"
df_ranked = pd.read_csv(ranking_path)

# Configure the Gemini API client using the API key from environment variables
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable not found. Please set it before running the application.")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def validate_input(text):
    if len(text) > 600:
        raise ValueError("The input is too long.")

    # Enhanced prompt for Gemini model
    prompt = f"""Determine if the following text is a valid hackathon or business idea. A valid business or hackathon idea usually describes a product, service, platform, or tool that solves a specific problem, provides a unique value to customers, or proposes an innovative solution. Respond with one of the following options:
    
    - 'yes' if it clearly describes a business or hackathon idea.
    - 'no' if it does not describe a business or hackathon idea.

Examples:
    1. Yes - “DoorDash - A platform that connects local restaurants with customers, offering seamless delivery and takeout options through an intuitive app, enabling individuals to enjoy a wide variety of cuisines from the comfort of their homes while helping small businesses reach a broader customer base and grow their revenue.”
    2. Yes - “EcoTrack - An app that allows individuals and companies to track their carbon footprint in real-time, offering personalized recommendations to reduce environmental impact and providing rewards for sustainable practices.”
    3. Yes - “Moocher - An app that allows college students to track free items and promotions offered on campus, fostering social interaction and a community-driven sharing economy.”
    4. Yes - “HealthBridge - A wearable device that monitors blood glucose levels and integrates with a mobile app, enabling diabetic patients to manage their condition effectively and share data with healthcare providers in real-time.”
    5. No - “The history of the Roman Empire and its impact on modern civilization.”
    6. No - “A collection of healthy recipes and workout tips for a balanced lifestyle.”
    7. No - “A blog about personal finance and budgeting.”

Text: “{text}”"""

    response = gemini_model.generate_content(prompt)
    answer = response.text.strip().lower()
    
    if 'yes' in answer:
        return True
    elif 'needs more information' in answer:
        raise ValueError("The input is too vague. Please provide more details about the hackathon or startup idea.")
    else:
        raise ValueError("The input does not appear to be a valid hackathon or startup idea.")
    

def normalize_input(text):
    try:
        title, description = text.split(" - ")
    except ValueError:
        raise ValueError("Invalid input format. Please provide the input as 'Title - Description'.")

    prompt = f"""
You are given information about a hackathon project or startup idea, including a title and a brief description. Your task is to reformat the description to be concise, professionally structured, and focused on explaining what the project does, its purpose, and any unique features it offers. Avoid using the title in the reformulated description.

Here are some examples to guide the format:

Input:
Title: "Flower"
Description: "Flower is an open-source framework for training AI on distributed data using federated learning. Companies like Banking Circle, Nokia, Porsche, and Brave use Flower to improve their AI models on sensitive data that is distributed across organizational silos or user devices."

Output:
"An open-source framework for federated learning, enabling companies to train AI on distributed, sensitive data across devices or organizational silos, unlocking new data potential in AI."

Input:
Title: "Trigo"
Description: "Trigo aggregates consumer rent history to help landlords approve better tenants and lenders write more loans. No solution exists to consistently furnish this data to landlords and lenders today. The largest database has only 3% coverage of rent data."

Output:
"Provides a comprehensive rent history database to support tenant approval and loan underwriting, addressing a critical gap in landlord and lender data access."

Input:
Title: "Scanbase"
Description: "Scanbase makes it easy for medical companies to convert photos of rapid diagnostic tests into results using a simple API."

Output:
"A straightforward API for medical companies to convert rapid diagnostic test images into digital results."

Now, reformat the following project information:

Title: "{title}"
Description: "{description}"

Reformatted Description:
"""
    
    response = gemini_model.generate_content(prompt)
    return f"{title} - {response.text.strip()}"

def get_score(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        score = torch.softmax(logits, dim=1)[0][1].item()  # Probability of 'good' class
    return score

def calculate_relative_rank(new_project_text):
    global df_ranked  # Declare df_ranked as global at the start
    
    # Validate input before proceeding
    validate_input(new_project_text)
    normalized_project_text = normalize_input(new_project_text)
    new_project_score = get_score(normalized_project_text)
    
    # Extract project name from the input text
    project_name = new_project_text.split(" - ")[0]

    # Check if a project with the same name already exists in the ranked DataFrame
    if project_name in df_ranked['text'].apply(lambda x: x.split(" - ")[0]).values:
        existing_project_index = df_ranked[df_ranked['text'].str.startswith(project_name)].index[0]
        existing_score = df_ranked.loc[existing_project_index, 'score']
        
        # Keep only the project with the higher score
        if new_project_score > existing_score:
            df_ranked.loc[existing_project_index, 'text'] = new_project_text
            df_ranked.loc[existing_project_index, 'score'] = new_project_score
            message = "Updated existing project with a higher score."
        else:
            message = "A project with this name already exists with a higher or equal score."
            return None, message
    else:
        # Append the new project if it doesn't already exist
        new_project_df = pd.DataFrame({'text': [new_project_text], 'score': [new_project_score]})
        df_ranked = pd.concat([df_ranked, new_project_df], ignore_index=True)

    # Sort and save the updated ranked projects
    df_sorted = df_ranked.sort_values(by='score', ascending=False).reset_index(drop=True)
    df_sorted[['text', 'score']].to_csv("ranked_projects.csv", index=False)
    
    # Calculate relative rank of the new project
    rank = df_sorted.index[df_sorted['text'] == new_project_text].tolist()[0]
    relative_rank = (df_sorted.shape[0] - rank) / df_sorted.shape[0]

    return relative_rank, message if 'message' in locals() else "New project added successfully."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rank', methods=['POST'])
def rank():
    data = request.get_json()
    new_project_text = data.get("idea")
    if new_project_text:
        try:
            relative_rank, message = calculate_relative_rank(new_project_text)
            response = {"relative_rank": relative_rank} if relative_rank is not None else {"message": message}
            return jsonify(response)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    return jsonify({"error": "Invalid input"}), 400

@app.route('/refine_pitch', methods=['POST'])
def refine_pitch():
    data = request.get_json()
    new_project_text = data.get("idea")
    if new_project_text:
        try:
            suggestion = normalize_input(new_project_text)
            return jsonify({"suggestion": suggestion})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    return jsonify({"error": "Invalid input"}), 400

@app.route('/leaderboard')
def leaderboard():
    # Load the ranked projects from the CSV
    df_ranked = pd.read_csv('ranked_projects.csv')
    
    # Sort by score in descending order to show top projects
    df_sorted = df_ranked.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # Calculate relative ranks (from 100% for the top project to ~0% for the bottom)
    df_sorted['relative_rank'] = ((len(df_sorted) - df_sorted.index) / len(df_sorted)) * 100
    
    # Select top 10 projects to display
    top_projects = df_sorted.head(10).to_dict(orient='records')
    
    return render_template('leaderboard.html', projects=top_projects)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT; default to 5000 if not set
    app.run(host="0.0.0.0", port=port)