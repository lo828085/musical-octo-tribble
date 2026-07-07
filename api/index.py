import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient

app = FastAPI()

# Allow your custom frontend UI to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hugging Face with your specific token
client = InferenceClient(
    api_key="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"
)

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        
        # Dynamically grab the requested model from the frontend, 
        # or default to Llama 3 8B if none is specified.
        selected_model = data.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
        
        # Hexon's core persona context
        messages = [
            {
                "role": "system",
                "content": "You are a custom AI named Hexon. You communicate with a concise, slightly futuristic tone. Do not use navigation bars in your text output."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        # Call the hosted open-source model
        response = client.chat.completions.create(
            model=selected_model, 
            messages=messages, 
            max_tokens=150
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
        # Queries Hugging Face for the top 200 text-generation models
        url = "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=200"
        response = requests.get(url)
        data = response.json()
        
        # Extract just the model IDs for your dropdown UI
        model_ids = [model["id"] for model in data]
        
        return {"total": len(model_ids), "models": model_ids}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
