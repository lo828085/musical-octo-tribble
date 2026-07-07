import os
import base64
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Your Base64 encoded token string
ENCODED_TOKEN = "aGZfcWRoZlpEWHZldFBNRmx0Q3RlaE96VlVkWXFucUhxQXd5RA=="

# 2. Decode the token
decoded_token = base64.b64decode(ENCODED_TOKEN).decode("utf-8")

# 3. Initialize Hugging Face
client = InferenceClient(
    api_key=decoded_token
)

# Safely locate index.html at the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
index_html_path = os.path.abspath(os.path.join(current_dir, "..", "index.html"))

@app.get("/")
async def serve_ui():
    """Serves the frontend UI directly at the root URL"""
    if os.path.exists(index_html_path):
        return FileResponse(index_html_path)
    return JSONResponse(
        status_code=404, 
        content={"error": f"index.html not found at expected path: {index_html_path}"}
    )

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        selected_model = data.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
        
        messages = [
            {
                "role": "system",
                "content": "You are Hexon, a highly capable AI assistant. Communicate concisely, accurately, and functionally. Format your output cleanly using markdown, especially for code blocks."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        # Call the hosted open-source model with fixed indentation
        response = client.chat.completions.create(
            model=selected_model, 
            messages=messages, 
            max_tokens=800
        )
        
        reply = response.choices[0].message.content
        return {
            "model_used": selected_model,
            "response": reply
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/models")
async def get_active_models():
    try:
        # Pushing the limit to 1000; sort=downloads ensures the best models are returned first
        url = "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=1000"
        response = requests.get(url)
        data = response.json()
        
        model_ids = [model["id"] for model in data]
        
        return {"total": len(model_ids), "models": model_ids}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
