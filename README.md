# OCR API with manga-ocr-base

FastAPI-based OCR service using the [manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) model from Hugging Face.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST /ocr-with-probabilities

Extract text from images with confidence scores and multiple predictions.

**Parameters:**
- `file` (required): Image file (JPEG, PNG, etc.)
- `num_predictions` (optional, default=3): Number of top predictions to return

**Response:**
```json
{
  "tokens": ["token1", "token2", ...],
  "probabilities": [0.95, 0.89, ...],
  "top_predictions": [
    {
      "text": "detected text version 1",
      "confidence": 0.92
    },
    {
      "text": "detected text version 2", 
      "confidence": 0.87
    }
  ]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## API Documentation

- Interactive docs: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

## Example Usage

```python
import requests

# Upload image for OCR
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr-with-probabilities",
        files={"file": f},
        data={"num_predictions": 5}
    )
    
result = response.json()
print(f"Top prediction: {result['top_predictions'][0]['text']}")
print(f"Confidence: {result['top_predictions'][0]['confidence']}")
```