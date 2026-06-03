import sys
try:
    from transformers import pipeline
    print("Transformers imported successfully")
    
    # Try loading biomedical-ner-all pipeline
    print("Attempting to load biomedical-ner-all...")
    pipe = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
    result = pipe("Patient presents with severe cough, fever, and chest pain.")
    print("Biomedical NER result:", result)
    
except Exception as e:
    print(f"Error occurred: {e}", file=sys.stderr)
