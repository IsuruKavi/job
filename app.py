from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    """
    Get BERT embeddings for a given text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def extract_skills(text):
    """
    Extract potential skills from text using a simple heuristic or pre-trained NER.
    """
    # Placeholder for actual skill extraction
    return set(token.lower() for token in text.split() if token.isalpha())

@app.route('/match_skills', methods=['POST'])
def match_skills():
    """
    API endpoint to extract skills from job description and resume,
    then calculate skill match similarity.
    """
    data = request.json
    job_description = data.get('job_description', '')
    resume_text = data.get('resume_text', '')

    # Extract skills from job description and resume
    job_skills = extract_skills(job_description)
    candidate_skills = extract_skills(resume_text)

    # Get BERT embeddings for the extracted skills
    job_skills_str = ' '.join(job_skills)
    candidate_skills_str = ' '.join(candidate_skills)

    job_embedding = get_bert_embeddings(job_skills_str)
    candidate_embedding = get_bert_embeddings(candidate_skills_str)

    # Calculate cosine similarity between job and candidate skill embeddings
    similarity = cosine_similarity([job_embedding], [candidate_embedding])[0][0] * 100

    return jsonify({
        'job_skills': list(job_skills),
        'candidate_skills': list(candidate_skills),
        'similarity': similarity
    })

if __name__ == '__main__':
    app.run(debug=True)
