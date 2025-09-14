import json
import logging
import os

from google import genai
from google.genai import types
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()
# IMPORTANT: KEEP THIS COMMENT
# Follow these instructions when using this blueprint:
# - Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
#   - do not change this unless explicitly requested by the user
# - Sometimes the google genai SDK has occasional type errors. You might need to run to validate, at time.  
# The SDK was recently renamed from google-generativeai to google-genai. This file reflects the new name and the new APIs.

# This API key is from Gemini Developer API Key, not vertex AI API Key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def summarize_article(text: str) -> str:
    prompt = f"Please summarize the following text concisely while maintaining key points:\n\n{text}"

    response = client.models.generate_content(model="gemini-2.5-flash",
                                              contents=prompt)

    return response.text or "SOMETHING WENT WRONG"


class Sentiment(BaseModel):
    rating: int
    confidence: float


def analyze_sentiment(text: str) -> Sentiment:
    try:
        system_prompt = (
            "You are a sentiment analysis expert. "
            "Analyze the sentiment of the text and provide a rating "
            "from 1 to 5 stars and a confidence score between 0 and 1. "
            "Respond with JSON in this format: "
            "{'rating': number, 'confidence': number}")

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Content(role="user", parts=[types.Part(text=text)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=Sentiment,
            ),
        )

        raw_json = response.text
        logging.info(f"Raw JSON: {raw_json}")

        if raw_json:
            data = json.loads(raw_json)
            return Sentiment(**data)
        else:
            raise ValueError("Empty response from model")

    except Exception as e:
        raise Exception(f"Failed to analyze sentiment: {e}")


def analyze_image(jpeg_image_path: str) -> str:
    with open(jpeg_image_path, "rb") as f:
        image_bytes = f.read()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                "Analyze this image in detail and describe its key " +
                "elements, context, and any notable aspects.",
            ],
        )

    return response.text if response.text else ""


def analyze_video(mp4_video_path: str) -> str:
    with open(mp4_video_path, "rb") as f:
        video_bytes = f.read()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=video_bytes,
                    mime_type="video/mp4",
                ),
                "Analyze this video in detail and describe its key " +
                "elements, context, and any notable aspects.",
            ],
        )

    return response.text if response.text else ""


def generate_image(prompt: str, image_path: str) -> None:
    response = client.models.generate_content(
        # IMPORTANT: only this gemini model supports image generation
        model="gemini-2.0-flash-preview-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']))

    if not response.candidates:
        return

    try:
        content = response.candidates[0].content
        if not content or not content.parts:
            return

        for part in content.parts:
            if part.text:
                print(part.text)
            elif part.inline_data and part.inline_data.data:
                with open(image_path, 'wb') as f:
                    f.write(part.inline_data.data)
                print(f"Image saved as {image_path}")
    except Exception as e:
        raise Exception(f"Failed to generate image: {e}")