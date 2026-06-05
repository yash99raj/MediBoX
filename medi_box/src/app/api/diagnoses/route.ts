import { NextRequest, NextResponse } from 'next/server';
import { 
  analyzeSymptomsWithGemini, 
  analyzeMedicalImageWithGemini,
  analyzeMedicalReportWithGemini,
  parseGeminiResponse,
  SymptomData,
  MedicalImageData
} from '@/lib/gemini';
import { 
  addDiagnosis, 
  getAllDiagnoses, 
  DiagnosisData 
} from '@/lib/store/diagnosesStore';

interface StructuredResponse {
  primaryDiagnosis: string;
  confidenceLevel: string;
  recommendations: string[];
  fullText: string;
  sections: string[];
}

export async function POST(request: NextRequest) {
  const ML_API_URL = process.env.ML_API_URL || "http://localhost:5001";
  try {
    const contentType = request.headers.get("content-type") || "";
    
    let type = "";
    let data: unknown = null;
    let fileToForward: File | null = null;
    
    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      type = (formData.get("type") as string) || "report";
      fileToForward = formData.get("file") as File;
    } else {
      const body = await request.json();
      type = body.type;
      data = body.data;
    }

    if (!type) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    let aiResultText = "";
    let structured: StructuredResponse = {
      primaryDiagnosis: "",
      confidenceLevel: "0",
      recommendations: [],
      fullText: "",
      sections: []
    };
    let extractedEntities: DiagnosisData['entities'] = undefined;
    
    let symptomData: SymptomData | null = null;
    let imageData: MedicalImageData | null = null;

    // Process based on diagnosis type
    if (type === 'symptoms') {
      symptomData = data as SymptomData;
      try {
        aiResultText = await analyzeSymptomsWithGemini(symptomData);
        structured = parseGeminiResponse(aiResultText);
        
        // Query local Flask NLP backend to extract entities from the symptoms text
        try {
          const flaskRes = await fetch(`${ML_API_URL}/api/symptoms`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symptoms: symptomData.description })
          });
          if (flaskRes.ok) {
            const flaskData = await flaskRes.json();
            extractedEntities = flaskData.entities;
          }
        } catch (err) {
          console.error("Failed to fetch Flask clinical entities for symptoms:", err);
        }
      } catch (geminiErr) {
        console.warn("Gemini symptoms analysis failed, falling back to local Flask models:", geminiErr);
        // Direct local fallback call to Flask
        const flaskRes = await fetch(`${ML_API_URL}/api/symptoms`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symptoms: symptomData.description })
        });
        if (!flaskRes.ok) throw new Error("Local Flask symptoms fallback failed");
        const flaskData = await flaskRes.json();
        
        aiResultText = `Local Clinical Analysis: ${flaskData.diagnosis}`;
        structured = {
          primaryDiagnosis: flaskData.diagnosis,
          confidenceLevel: String(flaskData.confidence),
          recommendations: flaskData.recommendations || [],
          fullText: aiResultText,
          sections: [
            `Local Analysis: ${flaskData.diagnosis}`,
            `Confidence: ${Number(flaskData.confidence || 0).toFixed(1)}%`
          ]
        };
        extractedEntities = flaskData.entities;
      }
    } 
    else if (type === 'image') {
      imageData = data as MedicalImageData;
      try {
        aiResultText = await analyzeMedicalImageWithGemini(imageData);
        structured = parseGeminiResponse(aiResultText);
        
        // Query local Flask NLP backend to extract entities from Gemini's findings
        try {
          const flaskRes = await fetch(`${ML_API_URL}/api/analyze-report`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: aiResultText })
          });
          if (flaskRes.ok) {
            const flaskData = await flaskRes.json();
            extractedEntities = flaskData.entities;
          }
        } catch (err) {
          console.error("Failed to fetch Flask clinical entities for image:", err);
        }
      } catch (geminiErr) {
        console.warn("Gemini image analysis failed, falling back to local Flask models:", geminiErr);
        
        // Build FormData to forward the base64 image data to the Flask predict endpoint
        try {
          const base64Clean = imageData.imageData.replace(/^data:image\/(png|jpeg|jpg);base64,/, "");
          const imgBuffer = Buffer.from(base64Clean, 'base64');
          
          const formData = new FormData();
          const fileBlob = new Blob([imgBuffer], { type: "image/jpeg" });
          formData.append("image", fileBlob, "image.jpg");
          
          const flaskRes = await fetch(`${ML_API_URL}/api/predict`, {
            method: "POST",
            body: formData
          });
          
          if (!flaskRes.ok) throw new Error(`Flask predict responded with: ${flaskRes.status}`);
          const flaskData = await flaskRes.json();
          
          aiResultText = `Local Medical Scan Analysis: ${flaskData.diagnosis}`;
          structured = {
            primaryDiagnosis: flaskData.diagnosis,
            confidenceLevel: String(flaskData.confidence),
            recommendations: [
              "Consult a qualified doctor to review the local image classification findings.",
              "Prepare relevant medical history for follow-up verification."
            ],
            fullText: aiResultText,
            sections: [
              `Local Scan Analysis: ${flaskData.diagnosis}`,
              `Confidence: ${Number(flaskData.confidence || 0).toFixed(1)}%`
            ]
          };
          
          // Try to extract entities from the diagnosis name
          try {
            const nerRes = await fetch(`${ML_API_URL}/api/analyze-report`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text: flaskData.diagnosis })
            });
            if (nerRes.ok) {
              const nerData = await nerRes.json();
              extractedEntities = nerData.entities;
            }
          } catch (nerErr) {
            console.error("NER extraction on local image diagnosis failed:", nerErr);
          }
        } catch (postErr) {
          console.error("Fallback image analysis pipeline failed:", postErr);
          throw postErr;
        }
      }
    }
    else if (type === 'report') {
      if (fileToForward) {
        const fileName = fileToForward.name.toLowerCase();
        const isImage = /\.(jpg|jpeg|png|webp|gif)$/.test(fileName);

        // Helper to detect report type from text or filename
        const detectReportType = (text: string): string => {
          const t = text.toLowerCase();
          if (t.includes("dengue")) return "Dengue Fever Panel";
          if (t.includes("haematology") || t.includes("hematology") || t.includes("cbc") || t.includes("complete blood count") || t.includes("hemoglobin") || t.includes("haemoglobin")) return "Complete Blood Count (CBC)";
          if (t.includes("thyroid") || t.includes("tsh") || t.includes("t3") || t.includes("t4")) return "Thyroid Function Panel";
          if (t.includes("lipid") || t.includes("cholesterol")) return "Lipid Profile";
          if (t.includes("liver") || t.includes("sgpt") || t.includes("sgot") || t.includes("bilirubin")) return "Liver Function Test (LFT)";
          if (t.includes("kidney") || t.includes("creatinine") || t.includes("urea")) return "Kidney Function Test (KFT)";
          if (t.includes("hba1c") || t.includes("glucose") || t.includes("diabetes")) return "Diabetes / Blood Sugar Panel";
          if (t.includes("malaria")) return "Malaria Panel";
          if (t.includes("typhoid") || t.includes("widal")) return "Typhoid Panel";
          if (t.includes("covid") || t.includes("sars-cov")) return "COVID-19 Panel";
          return "Medical Lab Report";
        };

        // Helper to extract diagnosis/recommendations from Gemini markdown response
        const extractFromGeminiMarkdown = (text: string) => {
          const diagMatch = text.match(/##\s*Primary Diagnosis\s*\n([^#]+)/i);
          const diagLine = diagMatch?.[1]?.trim().split('\n')[0].replace(/^[*-]\s*/, '') || "";

          const recMatch = text.match(/##\s*Recommended Next Steps\s*\n([^#]+)/i);
          const recommendations = recMatch?.[1]
            ? recMatch[1].split('\n').map((l: string) => l.trim().replace(/^[-*\d.]+\s*/, '')).filter((l: string) => l.length > 10 && l.length < 300)
            : [];

          const sectionMatches = text.match(/##\s+[^\n]+/g) || [];
          const sections = sectionMatches.map((s: string) => s.replace(/^##\s*/, ''));

          return { diagLine, recommendations, sections };
        };

        // ── IMAGE PATH: Send directly to Gemini Vision (bypasses EasyOCR) ──────
        if (isImage) {
          const fileBytes = await fileToForward.arrayBuffer();
          const base64 = Buffer.from(fileBytes).toString("base64");
          const mimeType = fileName.endsWith(".png") ? "image/png" : "image/jpeg";
          const reportType = detectReportType(fileName); // use filename as initial hint

          // Also run Flask for entity extraction (uses EasyOCR as secondary, non-blocking)
          let flaskEntities = undefined;
          let flaskLabData = {};
          let flaskSafetyWarning = null;
          try {
            const flaskFormData = new FormData();
            flaskFormData.append("file", fileToForward);
            const flaskRes = await fetch(`${ML_API_URL}/api/analyze-report`, { method: "POST", body: flaskFormData });
            if (flaskRes.ok) {
              const flaskData = await flaskRes.json();
              flaskEntities = flaskData.entities;
              flaskLabData = flaskData.lab_data || {};
              flaskSafetyWarning = flaskData.safety_warning || null;
            }
          } catch (err) {
            console.warn("Flask entity extraction failed (non-blocking):", err);
          }

          let geminiFullText = "";
          let geminiDiagnosis = "Pending AI analysis";
          let geminiConfidence = "72";
          let geminiRecommendations: string[] = [];
          let geminiSections: string[] = [];

          try {
            geminiFullText = await analyzeMedicalReportWithGemini({
              imageData: base64,
              imageMimeType: mimeType,
              labValues: flaskLabData,
              reportType
            });

            const { diagLine, recommendations, sections } = extractFromGeminiMarkdown(geminiFullText);
            if (diagLine.length > 5) geminiDiagnosis = diagLine;
            if (recommendations.length) geminiRecommendations = recommendations;
            if (sections.length) geminiSections = sections;
          } catch (err) {
            console.error("Gemini Vision report analysis failed", err);
            geminiFullText = "The AI could not process this image. Please ensure the image is clear and try again, or upload a PDF version.";
          }

          aiResultText = geminiFullText;
          structured = {
            primaryDiagnosis: geminiDiagnosis,
            confidenceLevel: geminiConfidence,
            recommendations: geminiRecommendations,
            fullText: aiResultText,
            sections: ["Gemini Vision Analysis", ...geminiSections]
          };

          if (flaskSafetyWarning) {
            structured.sections.unshift(`SAFETY WARNING: ${flaskSafetyWarning}`);
          }

          extractedEntities = flaskEntities;

        } else {
          // ── PDF/TEXT PATH: Forward to Flask OCR, then Gemini text ──────────
          const flaskFormData = new FormData();
          flaskFormData.append("file", fileToForward);

          const flaskRes = await fetch(`${ML_API_URL}/api/analyze-report`, {
            method: "POST",
            body: flaskFormData
          });

          if (!flaskRes.ok) {
            throw new Error(`Flask backend error: ${flaskRes.statusText}`);
          }

          const flaskData = await flaskRes.json();
          const ocrText = flaskData.text || "";

          let geminiFullText = "";
          let geminiDiagnosis = flaskData.diagnosis;
          let geminiConfidence = String(flaskData.confidence);
          let geminiRecommendations = flaskData.recommendations || [];
          let geminiSections: string[] = [];

          try {
            if (ocrText) {
              const reportType = detectReportType(ocrText);
              geminiFullText = await analyzeMedicalReportWithGemini({
                reportText: ocrText,
                labValues: flaskData.lab_data || {},
                reportType
              });

              const { diagLine, recommendations, sections } = extractFromGeminiMarkdown(geminiFullText);
              if (diagLine.length > 5 && diagLine !== "Refer to Detailed Assessment") geminiDiagnosis = diagLine;
              if (recommendations.length) geminiRecommendations = recommendations;
              if (sections.length) geminiSections = sections;
            }
          } catch (err) {
            console.error("Gemini report analysis failed", err);
            geminiFullText = "The report was analyzed by local models. The AI extracted key entities but a detailed generative summary could not be produced at this time. Please ensure your Gemini API key is valid.";
          }

          aiResultText = geminiFullText;
          structured = {
            primaryDiagnosis: geminiDiagnosis,
            confidenceLevel: geminiConfidence,
            recommendations: geminiRecommendations,
            fullText: aiResultText,
            sections: [
              `Local Clinical Model: ${flaskData.diagnosis}`,
              `Confidence: ${Number(flaskData.confidence || 0).toFixed(1)}%`,
              `Extracted Insights: ${flaskData.entities?.diseases?.length || 0} diseases, ${flaskData.entities?.medications?.length || 0} medications extracted.`,
              ...geminiSections
            ]
          };

          if (flaskData.safety_warning) {
            structured.sections.unshift(`SAFETY WARNING: ${flaskData.safety_warning}`);
          }
          if (flaskData.guidelines_retrieved) {
            structured.sections.push("Grounding: Medical guidelines were successfully retrieved for this analysis.");
          }

          extractedEntities = flaskData.entities;
        }

      } else if (data && typeof data === 'object' && 'text' in data) {
        const textReportData = data as { text: string };
        // Forward JSON text directly to Flask
        const flaskRes = await fetch(`${ML_API_URL}/api/analyze-report`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textReportData.text })
        });
        
        if (!flaskRes.ok) {
          throw new Error(`Flask backend error: ${flaskRes.statusText}`);
        }
        
        const flaskData = await flaskRes.json();
        const ocrText = flaskData.text || "";
        
        let geminiFullText = "";
        let geminiDiagnosis = flaskData.diagnosis;
        let geminiConfidence = String(flaskData.confidence);
        let geminiRecommendations = flaskData.recommendations || [];
        let geminiSections: string[] = [];

        // Auto-detect report type from text content
        const detectReportTypeInline = (text: string): string => {
          const t = text.toLowerCase();
          if (t.includes("dengue")) return "Dengue Fever Panel";
          if (t.includes("cbc") || t.includes("complete blood count")) return "Complete Blood Count (CBC)";
          if (t.includes("thyroid") || t.includes("tsh")) return "Thyroid Function Panel";
          if (t.includes("lipid") || t.includes("cholesterol")) return "Lipid Profile";
          if (t.includes("liver") || t.includes("sgpt") || t.includes("sgot")) return "Liver Function Test (LFT)";
          if (t.includes("kidney") || t.includes("creatinine")) return "Kidney Function Test (KFT)";
          if (t.includes("hba1c") || t.includes("glucose")) return "Diabetes / Blood Sugar Panel";
          return "Medical Lab Report";
        };

        try {
            if (ocrText) {
                const reportType = detectReportTypeInline(ocrText);
                const geminiAnalysis = await analyzeMedicalReportWithGemini({
                    reportText: ocrText,
                    labValues: flaskData.lab_data || {},
                    reportType
                });
                geminiFullText = geminiAnalysis;
                const diagMatch = geminiAnalysis.match(/##\s*Primary Diagnosis\s*\n([^#]+)/i);
                if (diagMatch?.[1]) {
                  const diagLine = diagMatch[1].trim().split('\n')[0].replace(/^[*-]\s*/, '');
                  if (diagLine.length > 5 && diagLine.length < 200) geminiDiagnosis = diagLine;
                }
                const recMatch = geminiAnalysis.match(/##\s*Recommended Next Steps\s*\n([^#]+)/i);
                if (recMatch?.[1]) {
                  geminiRecommendations = recMatch[1]
                    .split('\n')
                    .map((l: string) => l.trim().replace(/^[-*\d.]+\s*/, ''))
                    .filter((l: string) => l.length > 10 && l.length < 300);
                }
                const sectionMatches = geminiAnalysis.match(/##\s+[^\n]+/g) || [];
                if (sectionMatches.length > 0) geminiSections = sectionMatches.map((s: string) => s.replace(/^##\s*/, ''));
            }
        } catch (err) {
            console.error("Gemini report analysis failed", err);
            geminiFullText = "The report was analyzed by local models. The AI extracted key entities but a detailed generative summary could not be produced at this time. Please ensure your Gemini API key is valid.";
        }
        
        aiResultText = geminiFullText;
        structured = {
          primaryDiagnosis: geminiDiagnosis,
          confidenceLevel: geminiConfidence,
          recommendations: geminiRecommendations,
          fullText: aiResultText,
          sections: [
            `Local Clinical Model: ${flaskData.diagnosis}`,
            `Confidence: ${Number(flaskData.confidence || 0).toFixed(1)}%`
          ]
        }
        
        if (geminiSections.length > 0) {
            structured.sections = [...structured.sections, ...geminiSections];
        }
        
        if (flaskData.safety_warning) {
          structured.sections.unshift(`SAFETY WARNING: ${flaskData.safety_warning}`);
        }
        
        extractedEntities = flaskData.entities
      } else {
        return NextResponse.json(
          { error: "No report file or text provided" },
          { status: 400 }
        );
      }
    }
    else {
      return NextResponse.json(
        { error: "Invalid diagnosis type" },
        { status: 400 }
      );
    }
    
    // Create new diagnosis object
    const newDiagnosisData: Omit<DiagnosisData, 'id'> = {
      diagnosisDate: new Date().toISOString().split('T')[0],
      type: type === 'symptoms' ? 'Symptom Analysis' : type === 'image' ? `${imageData?.imageType || 'Image'} Analysis` : structured.sections?.[0]?.replace('Gemini Vision Analysis', 'Medical Report Analysis') || 'Medical Report Analysis',
      aiDiagnosis: structured.primaryDiagnosis || "Pending AI diagnosis",
      confidence: Math.round(parseFloat(structured.confidenceLevel)) || 0,
      status: "pending",
      symptoms: type === 'symptoms' ? (symptomData?.description || "") : type === 'report' ? "Parsed medical document file" : "Analyzed medical scan",
      doctorName: "Pending Review",
      doctorFeedback: "",
      imageSrc: type === 'image' 
        ? (imageData?.imageData || "") 
        : (fileToForward && /\.(jpg|jpeg|png|webp)$/i.test(fileToForward.name) 
            ? `data:image/jpeg;base64,${await (async () => { const b = await fileToForward.arrayBuffer(); return Buffer.from(b).toString('base64'); })()}` 
            : ""),
      aiModelData: {
        modelVersion: type === 'symptoms' ? "Gemini-Pro + ClinicalBERT" : type === 'image' ? "Gemini-Vision + ClinicalBERT" : "ClinicalBERT / BioBERT",
        analysisTimestamp: new Date().toISOString(),
        processingTime: "1.8 seconds",
        featuresAnalyzed: type === 'symptoms' ? "Hugging Face NER & Zero-Shot" : "Gemini Multimodal + HF NER"
      },
      treatmentRecommendations: structured.recommendations || [],
      riskFactors: [
        "To be determined by doctor review"
      ],
      aiResponse: {
        fullText: structured.fullText,
        sections: structured.sections
      },
      entities: extractedEntities
    };
    
    // Add the diagnosis to the store
    const newDiagnosis = addDiagnosis(newDiagnosisData);
    
    return NextResponse.json({ 
      success: true,
      diagnosisId: newDiagnosis.id,
      diagnosis: newDiagnosis
    });
  } catch (error) {
    console.error("Diagnosis creation error:", error);
    return NextResponse.json(
      { error: "Failed to create diagnosis" },
      { status: 500 }
    );
  }
}

// Get all diagnoses (paginated)
export async function GET(request: NextRequest) {
  try {
    // Get query parameters
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get('page') || '1');
    const limit = parseInt(searchParams.get('limit') || '10');
    const status = searchParams.get('status');
    
    // Get all diagnoses
    let diagnosisArray = getAllDiagnoses();
    
    // Apply filters
    if (status) {
      diagnosisArray = diagnosisArray.filter(d => d.status === status);
    }
    
    // Apply pagination
    const startIndex = (page - 1) * limit;
    const endIndex = page * limit;
    const paginatedDiagnoses = diagnosisArray.slice(startIndex, endIndex);
    
    return NextResponse.json({
      diagnoses: paginatedDiagnoses,
      total: diagnosisArray.length,
      page,
      limit,
      totalPages: Math.ceil(diagnosisArray.length / limit)
    });
  } catch (error) {
    console.error("API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch diagnoses" },
      { status: 500 }
    );
  }
} 