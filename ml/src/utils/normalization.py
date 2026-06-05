import re

# Canonical keys
NORMALIZATION_MAP = {
    # Haematology
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "hemoglobin": "hemoglobin",
    "plt": "platelet_count",
    "platelets": "platelet_count",
    "platelet count": "platelet_count",
    "wbc": "white_blood_cell_count",
    "white blood cells": "white_blood_cell_count",
    "total wbc count": "white_blood_cell_count",
    "rbc": "red_blood_cell_count",
    "red blood cells": "red_blood_cell_count",
    # Dengue-specific
    "igm": "igm",
    "igg": "igg",
    "dengue igm": "dengue_igm",
    "dengue igg": "dengue_igg",
    "dengue fever antibody, igm": "dengue_igm_elisa",
    "dengue fever antibody, igg": "dengue_igg_elisa",
    "dengue fever antibody igg elisa": "dengue_igg_elisa",
    "dengue fever antibody igm elisa": "dengue_igm_elisa",
    "dengue ns1 antigen": "dengue_ns1",
    # Metabolic
    "glucose": "glucose",
    "blood glucose": "glucose",
    "creatinine": "creatinine",
    "urea": "urea",
    "bilirubin": "bilirubin",
    "sgpt": "sgpt",
    "sgot": "sgot",
    "alt": "sgpt",
    "ast": "sgot",
    # Thyroid
    "tsh": "tsh",
    "t3": "t3",
    "t4": "t4",
    # Lipid
    "cholesterol": "total_cholesterol",
    "hdl": "hdl_cholesterol",
    "ldl": "ldl_cholesterol",
    "triglycerides": "triglycerides",
}

def normalize_key(key: str) -> str:
    """Normalize a lab test name into a standard canonical key."""
    clean_key = key.lower().strip()
    # Try exact match first
    if clean_key in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[clean_key]
    # Try partial match for long test names
    for map_key, norm_val in NORMALIZATION_MAP.items():
        if map_key in clean_key or clean_key in map_key:
            return norm_val
    return clean_key

def extract_lab_values_rule_based(text: str) -> dict:
    """
    Extract lab values using multiple regex patterns for structured lines.

    Pattern 1 (colon/equals style):
        'Hemoglobin: 7.1 g/dL' -> {'hemoglobin': '7.1'}
        'IgG = Positive' -> {'dengue_igg': 'Positive'}

    Pattern 2 (whitespace/tab separated - common in Dengue/lab panels):
        'DENGUE FEVER ANTIBODY, IgG ELISA   3.40   Positive'
        -> {'dengue_igg_elisa': '3.40 Positive'}

    Pattern 3 (inline positive/negative keyword):
        'IgM POSITIVE' or 'Dengue IgM: Positive'
    """
    results = {}
    lines = text.split('\n')

    # Pattern 1: Key: Value or Key = Value (with optional units)
    pattern1 = re.compile(
        r'^([a-zA-Z][a-zA-Z\s,\(\)]+?)[:=]\s*([\d\.]+|[Pp]ositive|[Nn]egative|[Ee]quivocal)',
        re.IGNORECASE
    )

    # Pattern 2: Whitespace-heavy lab rows (dengue panel style)
    # Captures test name (multi-word, may contain comma), numeric value, optional Pos/Neg
    pattern2 = re.compile(
        r'^([A-Z][A-Za-z\s,\(\)]+?)\s{2,}([\d\.]+)\s+(Positive|Negative|Equivocal)',
        re.IGNORECASE
    )

    # Pattern 3: Direct "Name Positive/Negative" on same line
    pattern3 = re.compile(
        r'(dengue\s+(?:fever\s+)?(?:antibody[,\s]+)?(?:igm|igg|ns1)[^\n]*?)\s+(positive|negative)',
        re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 4:
            continue

        # Try Pattern 2 first (more specific)
        m2 = pattern2.match(stripped)
        if m2:
            raw_key = m2.group(1).strip()
            numeric_val = m2.group(2).strip()
            qualitative = m2.group(3).strip()
            norm_key = normalize_key(raw_key)
            results[norm_key] = f"{numeric_val} ({qualitative})"
            continue

        # Try Pattern 1
        m1 = pattern1.match(stripped)
        if m1:
            raw_key = m1.group(1).strip()
            val = m1.group(2).strip()
            norm_key = normalize_key(raw_key)
            results[norm_key] = val
            continue

    # Scan full text for Pattern 3 (cross-line dengue detection)
    for m3 in pattern3.finditer(text):
        raw_key = m3.group(1).strip()
        val = m3.group(2).strip()
        norm_key = normalize_key(raw_key)
        if norm_key not in results:
            results[norm_key] = val

    return results
