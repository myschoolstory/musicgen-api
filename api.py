from transformers import AutoProcessor, MusicgenForConditionalGeneration
from fastapi import FastAPI, HTTPException
import modal

# Define the Modal app
app = modal.App("musicgen-api")

# Define the FastAPI app
fastapi_app = FastAPI()

# Load the Hugging Face model and processor
model_name = "facebook/musicgen-large"

@app.function()
@modal.web_endpoint()
def generate_music(prompt: str):
    """
    Generate music based on a text prompt using the facebook/musicgen-large model.
    """
    try:
        # Load the processor and model
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)

        # Prepare inputs for the model
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")

        # Generate audio values
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3,
            max_new_tokens=256
        )

        # Return the generated audio values
        return {"generated_audio": audio_values.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Expose FastAPI app as a Modal web endpoint
@app.asgi_app()
def asgi_app():
    return fastapi_app

# Deploy the app using `modal deploy`
