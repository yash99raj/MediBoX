import { GoogleGenerativeAI } from "@google/generative-ai";

// Initialize the Gemini API with your API key
// In production, use environment variables for API keys
const API_KEY = process.env.NEXT_PUBLIC_GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || "";
const genAI = new GoogleGenerativeAI(API_KEY);

export type SymptomData = {
  description: string;
  duration: string;
  severity: string;
  medicalHistory?: string;
};

export type MedicalImageData = {
  imageData: string; // Base64 encoded image
  imageType: string; // e.g., "X-ray", "MRI", "CT scan"
  bodyPart: string;
};

export type MedicalReportData = {
  reportText?: string;          // Full OCR-extracted text from the report (optional when image provided)
  imageData?: string;           // Base64 encoded image of the report (for Vision API)
  imageMimeType?: string;       // e.g., "image/jpeg", "image/png"
  labValues?: Record<string, string>; // Structured key-value pairs of lab results
  reportType?: string;         // e.g., "Blood Panel", "Dengue Panel", "CBC"
};

export async function analyzeSymptomsWithGemini(symptomData: SymptomData) {
  try {
    // For text-only generation, use the gemini-1.5-flash model
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

    // Create a structured prompt for symptom analysis
    const prompt = `
      You are a medical AI assistant. Analyze the following symptoms and provide a preliminary diagnosis.
      Please structure your response with the following sections:
      1. Potential Diagnoses (with confidence levels)
      2. Analysis Explanation
      3. Recommended Next Steps
      4. Risk Factors to Consider
      5. Treatment Suggestions (noting these are preliminary and subject to doctor confirmation)

      Patient Symptoms:
      Description: ${symptomData.description}
      Duration: ${symptomData.duration}
      Severity: ${symptomData.severity}
      ${symptomData.medicalHistory ? `Medical History: ${symptomData.medicalHistory}` : ''}
      
      Important: Always clarify that this is an AI-generated preliminary assessment and not a replacement for professional medical advice.
    `;

    // Generate content
    const result = await model.generateContent(prompt);
    const response = result.response;
    return response.text();
  } catch (error) {
    console.error("Error analyzing symptoms with Gemini:", error);
    throw error;
  }
}

export async function analyzeMedicalImageWithGemini(imageData: MedicalImageData) {
  try {
    // For multimodal generation (text + images), use the gemini-1.5-flash model
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

    // Decode the base64 image and create a part for the model
    const imagePart = {
      inlineData: {
        data: imageData.imageData.replace(/^data:image\/(png|jpeg|jpg);base64,/, ""),
        mimeType: imageData.imageData.includes("data:image/png") 
          ? "image/png" 
          : "image/jpeg",
      },
    };

    // Create a structured prompt for image analysis
    const prompt = `
      You are a medical imaging specialist AI. Analyze this ${imageData.imageType} of the ${imageData.bodyPart} 
      and provide a detailed assessment. Structure your response with:
      1. Key Findings
      2. Potential Diagnoses (with confidence levels)
      3. Notable Anomalies or Areas of Concern
      4. Comparison to Typical Results
      5. Recommendations for Further Analysis (if needed)
      
      Be specific about what you can see in the image and note any limitations in your analysis.
      Important: Clarify that this is an AI-generated preliminary assessment and not a replacement for professional medical diagnosis.
    `;

    // Generate content with the image
    const result = await model.generateContent([prompt, imagePart]);
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error("Error analyzing medical image with Gemini:", error);
    throw error;
  }
}

export async function getFollowUpRecommendations(diagnosis: string, patientQuestion?: string) {
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    
    const prompt = `
      You are a medical AI assistant helping with follow-up recommendations for a diagnosis.
      
      Diagnosis: ${diagnosis}
      ${patientQuestion ? `Patient Question: ${patientQuestion}` : ''}
      
      Please provide:
      1. Key follow-up actions the patient should take
      2. Warning signs that would require immediate medical attention
      3. Lifestyle recommendations related to this condition
      4. Questions the patient should ask their doctor
      
      Keep your response concise, informative, and focused on practical advice.
      Always clarify that this information is supplementary to professional medical advice.
    `;
    
    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error("Error getting follow-up recommendations with Gemini:", error);
    throw error;
  }
}

export async function analyzeMedicalReportWithGemini(reportData: MedicalReportData) {
  try {
    // Use the full flash model for better clinical reasoning on lab reports
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

    // Build a structured lab values section if available
    const labValuesSection = reportData.labValues && Object.keys(reportData.labValues).length > 0
      ? `\nExtracted Lab Values (structured):\n${Object.entries(reportData.labValues)
          .map(([k, v]) => `  - ${k}: ${v}`)
          .join("\n")}`
      : "";

    const clinicalPrompt = `You are a senior clinical laboratory specialist AI. You are analyzing a medical lab report. Your task is to produce a thorough, structured clinical interpretation.

Report Type: ${reportData.reportType || "Medical Lab Report"}
${labValuesSection}

Please carefully read ALL values from the report (including those in columns/tables) and provide a comprehensive analysis structured as follows:

## Primary Diagnosis
State the most likely diagnosis based on the lab findings. Be specific (e.g., "Dengue Fever — Positive IgG and IgM antibodies indicating Secondary Dengue Infection"). If all values are normal, state "No Significant Pathology Detected — All Parameters Within Normal Range".

## Key Lab Findings
For EACH test result in the report, provide:
- **Test Name:** Value (Unit) — Normal / Abnormal (High/Low) — Reference: [range]
- Clinical significance in 1 sentence

## Clinical Interpretation
A clear paragraph explaining what all results together indicate. Mention any patterns (e.g., lymphopenia, elevated MCHC) and their clinical relevance.

## Risk Assessment
List any abnormal findings that need attention, from most to least urgent.

## Recommended Next Steps
List 4-6 specific, actionable clinical recommendations.

## Important Disclaimer
This is an AI-assisted preliminary analysis and must be confirmed by a qualified physician.

Be thorough, evidence-based, and use plain language.`;

    // VISION PATH: When an image is provided, send directly to Gemini Vision (more accurate than OCR)
    if (reportData.imageData) {
      const imagePart = {
        inlineData: {
          data: reportData.imageData.replace(/^data:image\/(png|jpeg|jpg);base64,/, ""),
          mimeType: (reportData.imageMimeType || "image/jpeg") as "image/jpeg" | "image/png",
        },
      };
      const result = await model.generateContent([clinicalPrompt, imagePart]);
      return result.response.text();
    }

    // TEXT PATH: When OCR text is provided (e.g., from a PDF)
    const textPrompt = `${clinicalPrompt}

Full Report Text (OCR extracted):
---
${(reportData.reportText || "").substring(0, 4000)}
---`;

    const result = await model.generateContent(textPrompt);
    return result.response.text();
  } catch (error) {
    console.error("Error analyzing medical report with Gemini:", error);
    throw error;
  }
}

// Function to safely parse Gemini API responses into structured format
export function parseGeminiResponse(responseText: string) {
  try {
    // Split by sections (numbered lists or headings)
    const sections = responseText.split(/\n\d+\.|(?:\n|^)#+\s/).filter(Boolean);
    
    // Create structured response object 
    return {
      fullText: responseText,
      sections: sections.map((section) => section.trim()),
      primaryDiagnosis: extractDiagnosis(responseText),
      confidenceLevel: extractConfidence(responseText),
      recommendations: extractRecommendations(responseText),
    };
  } catch (error) {
    console.error("Error parsing Gemini response:", error);
    return { fullText: responseText, sections: [], primaryDiagnosis: "", confidenceLevel: "", recommendations: [] };
  }
}

// Helper functions to extract specific data from the response
function extractDiagnosis(text: string): string {
  // First look for bullet points or direct statements about diagnosis
  const diagnosisMatch = text.match(/(?:primary diagnosis|most likely diagnosis|potential diagnosis|suspected diagnosis|diagnosis)[:\-]?\s*\*?\*?\s*([^\n\.\*]+)/i);
  if (diagnosisMatch?.[1] && diagnosisMatch[1].trim().length > 0) {
    return diagnosisMatch[1].trim();
  }
  const fallbackMatch = text.match(/(?:diagnosis|diagnose) is ([^\n\.]+)/i);
  if (fallbackMatch?.[1] && fallbackMatch[1].trim().length > 0) {
    return fallbackMatch[1].trim();
  }
  
  // If we can't find a direct match, look for the first non-heading line under a Diagnosis section
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  
  // Find index of diagnosis section
  const diagSectionIdx = lines.findIndex(l => /diagnosis/i.test(l));
  if (diagSectionIdx !== -1 && diagSectionIdx + 1 < lines.length) {
      const possibleDiagLine = lines.slice(diagSectionIdx + 1).find(l => !l.match(/^#/));
      if (possibleDiagLine) {
          // clean up bullet points
          const cleanedLine = possibleDiagLine.replace(/^[-*\d.]+\s*/, '').replace(/\*+/g, '');
          if (cleanedLine.length > 0 && cleanedLine.length < 150) {
              return cleanedLine;
          }
      }
  }

  if (lines.length > 0) {
    const firstLine = lines.find(l => !l.startsWith('#') && l.length > 5 && l.length < 100);
    if (firstLine) {
      return firstLine.replace(/^[-*\d.]+\s*/, '').replace(/\*+/g, '');
    }
  }
  return "Refer to Detailed Assessment";
}

function extractConfidence(text: string): string {
  // Look for confidence levels
  const confidenceMatch = text.match(/(\d+(?:\.\d+)?\s*%)/);
  return confidenceMatch?.[1] || "";
}

function extractRecommendations(text: string): string[] {
  // Look for recommendation section and extract bullet points
  const recommendationsSection = text.match(/recommendations?[:\s]+([\s\S]+?)(?:\n\n|\n#|$)/i);
  if (!recommendationsSection) return [];
  
  return recommendationsSection[1]
    .split(/\n-|\n\*|\n•|\n\d+\./)
    .map(item => item.trim())
    .filter(Boolean);
} 