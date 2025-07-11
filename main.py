from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
from typing import List
from pydantic import BaseModel
import torch
import torch.nn.functional as F

class TextPrediction(BaseModel):
    text: str
    confidence: float

class OCRProbabilityResult(BaseModel):
    tokens: List[str]
    probabilities: List[float]
    top_predictions: List[TextPrediction]

app = FastAPI(title="OCR API", description="Image OCR using manga-ocr-base model")

processor = None
model = None

@app.on_event("startup")
async def startup_event():
    global processor, model
    try:
        processor = TrOCRProcessor.from_pretrained("kha-white/manga-ocr-base")
        model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.post("/ocr-with-probabilities", response_model=OCRProbabilityResult)
async def extract_text_with_probabilities(file: UploadFile = File(...), num_predictions: int = 3):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                num_beams=max(5, num_predictions),
                num_return_sequences=num_predictions,
                max_length=384
            )
        
        sequences = outputs.sequences
        scores = outputs.scores
        
        top_predictions = []
        for i, sequence in enumerate(sequences):
            text = processor.decode(sequence, skip_special_tokens=True)
            
            if scores:
                sequence_score = 0.0
                for step_scores in scores:
                    if i < len(step_scores):
                        probs = F.softmax(step_scores[i], dim=-1)
                        max_prob = torch.max(probs).item()
                        sequence_score += max_prob
                
                avg_confidence = sequence_score / len(scores) if scores else 0.0
            else:
                avg_confidence = 0.0
            
            top_predictions.append(TextPrediction(
                text=text,
                confidence=avg_confidence
            ))
        
        tokens = []
        probabilities = []
        
        if scores:
            for step_scores in scores:
                probs = F.softmax(step_scores[0], dim=-1)
                top_prob_idx = torch.argmax(probs).item()
                top_prob_value = probs[top_prob_idx].item()
                
                token = processor.tokenizer.decode([top_prob_idx])
                tokens.append(token)
                probabilities.append(top_prob_value)
        
        return OCRProbabilityResult(
            tokens=tokens,
            probabilities=probabilities,
            top_predictions=top_predictions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)