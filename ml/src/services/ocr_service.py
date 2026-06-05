import os

# Try to import pdfplumber and paddleocr, fail gracefully if not installed
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import easyocr
    # Initialize EasyOCR globally so we don't reload it every time
    # Use CPU for default.
    ocr = easyocr.Reader(['en'], gpu=False)
except ImportError:
    ocr = None

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text and tables from a PDF using pdfplumber."""
    if not pdfplumber:
        return "pdfplumber is not installed. Please install it to process PDFs."
        
    extracted_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract basic text
            text = page.extract_text()
            if text:
                extracted_text.append(text)
            
            # Extract tables as structured text representation
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    # Clean up empty cells
                    row_text = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                    if any(row_text):
                        extracted_text.append(" | ".join(row_text))
                        
    return "\n".join(extracted_text)

def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image using EasyOCR."""
    if not ocr:
        return "EasyOCR is not installed. Please install it to process Images."
        
    result = ocr.readtext(file_path)
    extracted_text = []
    
    for item in result:
        extracted_text.append(item[1])
            
    return "\n".join(extracted_text)

def process_document(file_path: str, file_extension: str) -> str:
    """Route document to correct OCR engine based on extension."""
    ext = file_extension.lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    else:
        # Fallback for plain text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return f"Unsupported file type: {ext}"
