# Hackathon Project Idea Ranking App

This application ranks hackathon and startup project ideas by analyzing their descriptions through a fine-tuned RoBERTa model. It offers relative ranking scores for submissions, a leaderboard of top ideas, and pitch refinement suggestions. The goal is to provide a valuable tool for entrepreneurs and hackathon participants to gauge the quality of their project ideas and improve them accordingly.

## Motivation and Approach

The app was created to provide quick, insightful feedback on project ideas. Many hackathon participants and budding entrepreneurs lack immediate feedback on whether their ideas are compelling and feasible. This app bridges that gap, drawing from real data on both successful and unsuccessful projects.

## Fine-Tuning the Model

### Dataset

The model was fine-tuned using two distinct datasets:

1. **Positive Examples**: Startups from the last four years sourced from Y Combinator (YC), representing successful projects with well-defined problems, clear value propositions, and compelling pitches.
2. **Negative Examples**: Unsuccessful projects from RevolutionUC (the largest in-person hackathon at the University of Cincinnati), spanning the last eight years. These represent ideas that didn’t resonate as strongly with judges.

By combining YC startups and less successful RevolutionUC projects, the model gained a balanced understanding of “good” versus “bad” project ideas based on real-world outcomes.

### Data Augmentation and Normalization

To make the data more consistent and help the model focus on the content of ideas rather than stylistic differences, I implemented the following:

- **Normalization**: GPT-4 was used to reformat and standardize project descriptions, making each idea structurally similar. This ensured that the model focused on content quality rather than language style, which prevented it from overfitting on pitch formats specific to either students or seasoned entrepreneurs.
  
- **Augmentation**: GPT-4 was also used to generate synthetic data, primarily for creating high-quality negative examples. While creating synthetic positive examples was challenging and often ineffective, GPT-4 generated negative examples that sounded promising but were conceptually flawed. This helped the model distinguish substance from superficial appeal.

This approach enabled the model to learn not only what makes a good idea but also what makes a poor one, especially when it appears compelling at first glance.

### Scraping and Data Collection

To compile the training data:

1. **YC Startup Scraper**: A scraper was used to gather data on YC startups from the past four years, capturing project names and descriptions for a dataset of compelling startup ideas.
2. **RevolutionUC Submissions**: As the director of RevolutionUC, I had access to historical submission data from RevolutionUC hackathons going back to 2015. This provided an authentic set of negative examples without needing to scrape public Devpost data, focusing on project descriptions and omitting any personally identifiable information.

Each submission was then categorized based on its success level, forming a labeled dataset of positive and negative examples.

## Application Features

- **Project Ranking**: Users can submit a project idea to receive a relative ranking score based on quality and feasibility.
- **Leaderboard**: Displays the top-ranked project ideas based on model scoring, showcasing what makes a project stand out.
- **Pitch Refinement**: Users can refine their pitch with AI-generated suggestions, allowing them to reformat their idea into a more compelling description.

## Technical Details

### Machine Learning Model

The model is a fine-tuned version of RoBERTa for sequence classification, trained to analyze project descriptions and assign relative quality scores. It was trained on a mix of positive (YC startups) and negative (RevolutionUC) examples, with data augmentation applied for enhanced understanding.

### API Endpoints

- `POST /rank`: Receives a project idea, ranks it, and returns a relative ranking score.
- `POST /refine_pitch`: Suggests improvements for a project description by generating a refined pitch.
- `GET /leaderboard`: Displays the top-ranked project ideas.

### Data Flow and Storage

Project scores are stored in `ranked_projects.csv`, which dynamically updates as new submissions are scored. Only unique entries are kept, with the highest-scoring version of each project retained.

## Installation

### Requirements

- Python 3.8+
- Google Gemini API access with an API key
- Required Python packages (listed in `requirements.txt`)

### Running the Model Fine-Tuning

Before using the app, you need to run the `model.py` script to fine-tune the model on YC startups and RevolutionUC hackathon projects. This prepares the model for ranking and evaluating new project ideas.

#### Fine-Tuning Steps

1. Run `model.py` in your project directory.
2. After fine-tuning, the model will be saved in the specified directory for use by the app.

### Setup Steps

1. Clone the repository.
2. Set up a virtual environment.
3. Install dependencies.
4. Set up your environment variables, including `GEMINI_API_KEY`.
5. Run `model.py` to fine-tune RoBERTa, which should only take a few minutes.
6. Run the Flask application and access it at `http://127.0.0.1:5000/`.

### Example Usage

1. **Submit a Project**: Enter a project name and description to receive a ranking.
2. **Refine Your Pitch**: Click “Refine My Pitch” to receive suggestions on improving the idea.
3. **View Leaderboard**: Navigate to the leaderboard to view the top project ideas.

## Conclusion

This project provides a tool for hackathon participants, entrepreneurs, and innovators to understand what makes a project idea stand out. By combining real-world examples of success and failure with insightful feedback, it helps users refine their concepts and presents a model for evaluating project quality in a practical, scalable way.

## License

This project is licensed under the MIT License.