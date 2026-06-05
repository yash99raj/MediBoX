def run_safety_checks(lab_data: dict) -> dict:
    """
    Run deterministic safety checks on normalized lab data.
    Returns a warning payload if a critical threshold is breached.
    Covers: platelets, hemoglobin, dengue antibodies, blood glucose, and creatinine.
    """
    warnings = []
    severity = "Warning"

    # --- DENGUE SAFETY CHECKS ---
    # Both IgG and IgM positive = Secondary Dengue (higher risk of severe dengue)
    dengue_igg = (
        lab_data.get("dengue_igg_elisa")
        or lab_data.get("dengue_igg")
        or lab_data.get("igg", "")
    )
    dengue_igm = (
        lab_data.get("dengue_igm_elisa")
        or lab_data.get("dengue_igm")
        or lab_data.get("igm", "")
    )

    igg_positive = "positive" in str(dengue_igg).lower()
    igm_positive = "positive" in str(dengue_igm).lower()

    if igg_positive and igm_positive:
        warnings.append(
            "Both Dengue IgG and IgM antibodies are POSITIVE — findings are consistent with "
            "Secondary Dengue Infection. Secondary dengue carries elevated risk of Dengue Haemorrhagic "
            "Fever (DHF) and Dengue Shock Syndrome (DSS). Immediate medical evaluation required."
        )
        severity = "Critical"
    elif igm_positive and not igg_positive:
        warnings.append(
            "Dengue IgM antibody is POSITIVE (IgG negative) — findings suggest Primary Dengue Infection. "
            "Monitor closely for warning signs including sudden platelet drop, bleeding, severe abdominal pain."
        )
        severity = "High"
    elif igg_positive and not igm_positive:
        warnings.append(
            "Dengue IgG antibody is POSITIVE (IgM negative) — may indicate past Dengue exposure or early "
            "secondary infection. Repeat testing recommended if symptoms are present."
        )
        severity = "Moderate"

    # --- PLATELET SAFETY CHECKS ---
    if "platelet_count" in lab_data:
        try:
            plt_str = str(lab_data["platelet_count"]).replace(",", "").split()[0]
            plt = float(plt_str)
            if plt < 20000:
                warnings.append(
                    f"CRITICAL: Platelet count is dangerously low ({plt:,.0f}/μL < 20,000). "
                    "Severe bleeding risk. Immediate hospitalisation and possible platelet transfusion required."
                )
                severity = "Critical"
            elif plt < 50000:
                warnings.append(
                    f"Platelet count is severely low ({plt:,.0f}/μL). "
                    "High risk of spontaneous bleeding. Hospital admission strongly recommended."
                )
                severity = "Critical"
            elif plt < 100000:
                warnings.append(
                    f"Platelet count is low ({plt:,.0f}/μL). Monitor daily — urgent repeat CBC recommended."
                )
                if severity not in ("Critical",):
                    severity = "High"
        except (ValueError, IndexError):
            pass

    # --- HEMOGLOBIN SAFETY CHECKS ---
    if "hemoglobin" in lab_data:
        try:
            hb = float(str(lab_data["hemoglobin"]).split()[0])
            if hb < 6.5:
                warnings.append(
                    f"CRITICAL: Hemoglobin is dangerously low ({hb} g/dL). "
                    "Severe anaemia — risk of cardiac stress and organ damage. Immediate medical attention required."
                )
                severity = "Critical"
            elif hb < 8.0:
                warnings.append(
                    f"Hemoglobin is significantly low ({hb} g/dL). "
                    "Moderate-severe anaemia. Consider further evaluation and possible transfusion."
                )
                if severity not in ("Critical",):
                    severity = "High"
        except (ValueError, IndexError):
            pass

    # --- BLOOD GLUCOSE CHECKS ---
    if "glucose" in lab_data:
        try:
            glc = float(str(lab_data["glucose"]).split()[0])
            if glc > 400:
                warnings.append(
                    f"Blood glucose is critically elevated ({glc} mg/dL). "
                    "Risk of diabetic ketoacidosis. Immediate medical care required."
                )
                severity = "Critical"
            elif glc < 50:
                warnings.append(
                    f"Blood glucose is critically low ({glc} mg/dL). "
                    "Severe hypoglycaemia — risk of loss of consciousness. Immediate sugar intake and medical attention needed."
                )
                severity = "Critical"
        except (ValueError, IndexError):
            pass

    # --- CREATININE CHECKS ---
    if "creatinine" in lab_data:
        try:
            cr = float(str(lab_data["creatinine"]).split()[0])
            if cr > 5.0:
                warnings.append(
                    f"Creatinine is critically elevated ({cr} mg/dL). "
                    "Possible acute kidney failure. Immediate nephrology consultation required."
                )
                severity = "Critical"
        except (ValueError, IndexError):
            pass

    if warnings:
        return {
            "severity": severity,
            "recommendation": (
                "⚠️ Seek immediate medical attention. These laboratory findings require urgent clinical review. "
                "Do not self-medicate — consult a qualified physician immediately."
            ),
            "warning": " | ".join(warnings)
        }

    return None
