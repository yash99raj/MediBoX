from flask import Flask, request, jsonify
import numpy as np
import os
import json
from PIL import Image
import io
import pypdf
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    pipeline = None
    transformers_available = False
    print("Warning: Hugging Face transformers/torch not installed. NLP will run in rule-based fallback mode.")

try:
    import tensorflow as tf
    from models.model import MedicalImageClassifier
    tensorflow_available = True
except ImportError:
    tf = None
    MedicalImageClassifier = None
    tensorflow_available = False
    print("Warning: TensorFlow is not installed. Image classification will run in mock mode.")

app = Flask(__name__)

# Lazy-load Hugging Face clinical models to keep Flask startup fast and robust
ner_pipeline = None
classifier_pipeline = None
models_loaded = False

CANDIDATE_MAP = {
    "lung or respiratory issue": "Respiratory Infection",
    "headache or migraine": "Migraine / Headache",
    "heart or chest pain issue": "Cardiac Condition",
    "stomach or stomach ache": "Gastrointestinal Issue",
    "neurological problem": "Neurological Disorder",
    "healthy or normal": "Normal / Healthy"
}

def load_models():
    global ner_pipeline, classifier_pipeline, models_loaded
    if models_loaded:
        return True
    if not transformers_available or pipeline is None:
        return False
    try:
        print("Loading Hugging Face clinical models (this may take a minute on first run)...")
        # Load NER pipeline
        ner_pipeline = pipeline(
            "ner", 
            model="d4data/biomedical-ner-all", 
            aggregation_strategy="simple"
        )
        # Load Zero-Shot Classification pipeline
        classifier_pipeline = pipeline(
            "zero-shot-classification", 
            model="typeform/distilbert-base-uncased-mnli"
        )
        models_loaded = True
        print("Hugging Face clinical models loaded successfully.")
        return True
    except Exception as e:
        print(f"Warning: Failed to load Hugging Face models ({e}). Using rule-based fallback.")
        return False

# Rule-based fallback for entity extraction
def fallback_ner(text):
    text_lower = text.lower()
    entities = {
        "symptoms": [],
        "medications": [],
        "diseases": [],
        "procedures": []
    }
    
    symptoms_list = ["cough", "fever", "chest pain", "shortness of breath", "headache", "nausea", "fatigue", "dizziness", "sore throat"]
    for s in symptoms_list:
        if s in text_lower:
            entities["symptoms"].append(s)
            
    meds_list = ["aspirin", "ibuprofen", "amoxicillin", "albuterol", "paracetamol", "acetaminophen", "metformin", "lisinopril"]
    for m in meds_list:
        if m in text_lower:
            entities["medications"].append(m)
            
    diseases_list = ["pneumonia", "covid-19", "tuberculosis", "lung cancer", "migraine", "bronchitis", "asthma", "diabetes"]
    for d in diseases_list:
        if d in text_lower:
            entities["diseases"].append(d)
            
    proc_list = ["x-ray", "mri", "ct scan", "blood test", "biopsy", "ultrasound"]
    for p in proc_list:
        if p in text_lower:
            entities["procedures"].append(p)
            
    return entities

# Rule-based fallback for classification
def fallback_classification(text):
    text_lower = text.lower()
    if "cough" in text_lower or "pneumonia" in text_lower or "covid" in text_lower:
        return "Respiratory Infection", 75.0
    if "headache" in text_lower or "migraine" in text_lower:
        return "Migraine / Headache", 70.0
    if "chest pain" in text_lower or "cardiac" in text_lower:
        return "Cardiac Condition", 80.0
    if "abdominal" in text_lower or "stomach" in text_lower or "nausea" in text_lower:
        return "Gastrointestinal Issue", 65.0
    return "Inconclusive Analysis", 50.0

def expand_to_word_boundaries(text, start, end):
    # Expand start backwards to find start of word
    while start > 0 and (text[start-1].isalnum() or text[start-1] == "-"):
        start -= 1
    # Expand end forwards to find end of word
    while end < len(text) and (text[end].isalnum() or text[end] == "-"):
        end += 1
    return text[start:end]

def extract_clinical_entities(text):
    if not load_models() or ner_pipeline is None:
        return fallback_ner(text)
    
    try:
        results = ner_pipeline(text)
        entities = {
            "symptoms": set(),
            "medications": set(),
            "diseases": set(),
            "procedures": set()
        }
        
        # Merge sequential subword tokens (e.g. 'am' + '##oxicillin' -> 'amoxicillin')
        merged_results = []
        current_item = None
        
        for item in results:
            word = item.get("word", "")
            grp = item.get("entity_group", "")
            start = item.get("start", 0)
            end = item.get("end", 0)
            
            if current_item is None:
                current_item = {"entity_group": grp, "word": word, "start": start, "end": end}
            else:
                # Merge if same entity group AND adjacent/subword in text
                if grp == current_item["entity_group"] and start == current_item["end"]:
                    current_item["word"] = current_item["word"] + word.lstrip("##")
                    current_item["end"] = end
                elif grp == current_item["entity_group"] and word.startswith("##"):
                    current_item["word"] = current_item["word"] + word[2:]
                    current_item["end"] = end
                else:
                    merged_results.append(current_item)
                    current_item = {"entity_group": grp, "word": word, "start": start, "end": end}
        if current_item:
            merged_results.append(current_item)
            
        # Process and clean up merged results
        for item in merged_results:
            grp = item.get("entity_group", "")
            start = item.get("start", 0)
            end = item.get("end", 0)
            
            # Reconstruct the full word from the original source text boundaries
            word = expand_to_word_boundaries(text, start, end).strip()
            word = word.replace(" - ", "-")  # Clean up 'x - ray' -> 'x-ray'
            if not word or len(word) < 2:
                continue
            
            if grp == "Sign_symptom":
                entities["symptoms"].add(word.lower())
            elif grp == "Medication":
                entities["medications"].add(word.lower())
            elif grp == "Disease_disorder":
                entities["diseases"].add(word.lower())
            elif grp in ["Diagnostic_procedure", "Therapeutic_procedure"]:
                entities["procedures"].add(word.lower())
        
        return {k: sorted(list(v)) for k, v in entities.items()}
    except Exception as e:
        print(f"NER extraction error: {e}")
        return fallback_ner(text)

def classify_clinical_text(text):
    if not load_models() or classifier_pipeline is None:
        return fallback_classification(text)
    
    try:
        candidate_labels = list(CANDIDATE_MAP.keys())
        result = classifier_pipeline(
            text, 
            candidate_labels=candidate_labels,
            hypothesis_template="This patient report indicates a {}."
        )
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        if labels and scores:
            best_label = labels[0]
            mapped_diagnosis = CANDIDATE_MAP[best_label]
            return mapped_diagnosis, float(scores[0] * 100)
        return "Inconclusive Analysis", 50.0
    except Exception as e:
        print(f"Classification error: {e}")
        return fallback_classification(text)

def generate_recommendations(diagnosis, entities):
    recs = []
    if diagnosis == "Respiratory Infection":
        recs.append("Rest and maintain adequate hydration.")
        recs.append("Monitor body temperature and oxygen levels regularly.")
        recs.append("Consult a doctor if shortness of breath develops.")
    elif diagnosis == "Migraine / Headache":
        recs.append("Rest in a quiet, dark room.")
        recs.append("Avoid known triggers like bright screens, caffeine, or stress.")
        recs.append("Consult a doctor if headaches become frequent or severe.")
    elif diagnosis == "Cardiac Condition":
        recs.append("Avoid strenuous physical exertion.")
        recs.append("Seek immediate emergency medical care if chest pain intensifies or spreads.")
        recs.append("Consult a cardiologist for complete diagnostic evaluation.")
    elif diagnosis == "Gastrointestinal Issue":
        recs.append("Eat a bland diet and stay hydrated.")
        recs.append("Avoid spicy, greasy, or acidic foods.")
    else:
        recs.append("Schedule a comprehensive review with your primary care physician.")
    
    if entities.get("medications"):
        recs.append(f"Ensure all listed medications ({', '.join(entities['medications'])}) are reviewed by a doctor for potential interactions.")
        
    return recs

def extract_text_from_pdf(file_bytes):
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

# Load the image classifier model - in production, you would load a pre-trained model
model = None

# Define class labels for images
CLASS_LABELS = [
    "Pneumonia",
    "Normal",
    "COVID-19",
    "Tuberculosis",
    "Lung Cancer"
]

@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Medical Image Analysis API"})

@app.route("/api/predict", methods=["POST"])
def predict():
    # Check if image was provided
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    try:
        # Read image
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Preprocess the image
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        if model is None:
            # Mock prediction for demonstration
            confidence_scores = np.random.random(len(CLASS_LABELS))
            confidence_scores = confidence_scores / np.sum(confidence_scores)
            prediction_idx = np.argmax(confidence_scores)
        else:
            # Real prediction
            predictions = model.predict(img_array)
            prediction_idx = np.argmax(predictions[0])
            confidence_scores = predictions[0]
        
        # Return result
        result = {
            "diagnosis": CLASS_LABELS[prediction_idx],
            "confidence": float(confidence_scores[prediction_idx] * 100),
            "all_probabilities": {
                label: float(confidence_scores[i] * 100) 
                for i, label in enumerate(CLASS_LABELS)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/symptoms", methods=["POST"])
def analyze_symptoms():
    # Get symptoms from request
    data = request.json
    if not data or "symptoms" not in data:
        return jsonify({"error": "No symptoms provided"}), 400
    
    symptoms = data["symptoms"]
    
    # Analyze symptoms using the clinical NLP model
    entities = extract_clinical_entities(symptoms)
    diagnosis, confidence = classify_clinical_text(symptoms)
    recommendations = generate_recommendations(diagnosis, entities)
    
    return jsonify({
        "diagnosis": diagnosis,
        "confidence": confidence,
        "entities": entities,
        "recommendations": recommendations
    })

@app.route("/api/analyze-report", methods=["POST"])
def analyze_report():
    text = ""
    
    # Check if a file was uploaded
    if "file" in request.files:
        file = request.files["file"]
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file.read())
        elif filename.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        else:
            return jsonify({"error": "Unsupported file format. Please upload a PDF or TXT file."}), 400
    
    # Check if text was provided in the JSON body
    elif request.is_json:
        data = request.json
        text = data.get("text", "")
        
    if not text.strip():
        return jsonify({"error": "No medical text or file content could be extracted."}), 400
        
    # Analyze the clinical text
    entities = extract_clinical_entities(text)
    diagnosis, confidence = classify_clinical_text(text)
    recommendations = generate_recommendations(diagnosis, entities)
    
    response = {
        "text": text[:5000], # Return truncated source text for display
        "diagnosis": diagnosis,
        "confidence": confidence,
        "entities": entities,
        "recommendations": recommendations
    }
    
    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False) 