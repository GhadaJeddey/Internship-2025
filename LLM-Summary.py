import json 
import requests 
import os 
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from groq import Groq

def groq_summarize_article(article_text):
    client = Groq(
        api_key="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",
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following text in one clear and concise paragraph, capturing the key ideas without missing critical points. Ensure the summary is easy to understand and avoids excessive detail. : {article_text}",
            }
        ],
        model="LLaMA-3.3-70B-Versatile",
    )

    summary = chat_completion.choices[0].message.content
    print("Summary:", summary)
    return summary

def gemini_summarize_article(article_text):
    genai.configure(api_key="AIzaSyC1OVCakG2ug2Bz5sKgJLTujr5mguPtHU0")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    response = model.generate_content(
        f"Summarize the following text in one clear and concise paragraph, capturing the key ideas without missing critical points. Ensure the summary is easy to understand and avoids excessive detail: {article_text}",
        generation_config=GenerationConfig(max_output_tokens=500)
    )

    summary = response.text
    print("Summary:", summary)
    return summary

with open('articles.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

article_key = "Medium-1" 
article_text = articles.get(article_key)
groq_summarize_article(article_text)
#gemini_summarize_article(article_text) 

