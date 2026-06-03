import { NextRequest, NextResponse } from 'next/server';
import { 
  analyzeSymptomsWithGemini, 
  analyzeMedicalImageWithGemini,
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
            `Confidence: ${(flaskData.confidence ?? 0).toFixed(1)}%`
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
              `Confidence: ${(flaskData.confidence ?? 0).toFixed(1)}%`
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
        // Forward the PDF/TXT file directly to the Flask server
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
        
        aiResultText = flaskData.text || "Report analysis completed by local models.";
        structured = {
          primaryDiagnosis: flaskData.diagnosis,
          confidenceLevel: String(flaskData.confidence),
          recommendations: flaskData.recommendations || [],
          fullText: aiResultText,
          sections: [
            `Local Clinical Model: ${flaskData.diagnosis}`,
            `Confidence: ${(flaskData.confidence ?? 0).toFixed(1)}%`,
            `Extracted Insights: ${flaskData.entities?.diseases?.length || 0} diseases, ${flaskData.entities?.medications?.length || 0} medications extracted.`
          ]
        };
        extractedEntities = flaskData.entities;
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
        aiResultText = flaskData.text;
        structured = {
          primaryDiagnosis: flaskData.diagnosis,
          confidenceLevel: String(flaskData.confidence),
          recommendations: flaskData.recommendations || [],
          fullText: aiResultText,
          sections: [
            `Local Clinical Model: ${flaskData.diagnosis}`,
            `Confidence: ${(flaskData.confidence ?? 0).toFixed(1)}%`
          ]
        };
        extractedEntities = flaskData.entities;
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
      type: type === 'symptoms' ? 'Symptom Analysis' : type === 'image' ? `${imageData?.imageType || 'Image'} Analysis` : 'Medical Report Analysis',
      aiDiagnosis: structured.primaryDiagnosis || "Pending AI diagnosis",
      confidence: Math.round(parseFloat(structured.confidenceLevel)) || 0,
      status: "pending",
      symptoms: type === 'symptoms' ? (symptomData?.description || "") : type === 'report' ? "Parsed medical document file" : "Analyzed medical scan",
      doctorName: "Pending Review",
      doctorFeedback: "",
      imageSrc: type === 'image' ? (imageData?.imageData || "") : "",
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