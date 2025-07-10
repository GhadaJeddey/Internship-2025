import json 
import requests 
import os 
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from groq import Groq

def groq_summarize_article(article_text,model,prompt):
    client = Groq(
        api_key="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",
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": str(prompt) + f" .Article : {article_text} ", 
            }
        ],
        model=model,
    )
    
    summary = chat_completion.choices[0].message.content
    print("Summary:", summary)
    return summary

def gemini_summarize_article(article_text,prompt):
    genai.configure(api_key="AIzaSyC1OVCakG2ug2Bz5sKgJLTujr5mguPtHU0")
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    response = model.generate_content(
        str(prompt) + f" .Article : {article_text}",
        generation_config=GenerationConfig(max_output_tokens=500)
    )

    summary = response.text
    print("Summary:", summary)
    return summary

with open('articles.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

article_key = "Long-2"
article_text = articles.get(article_key)

prompt_gemini = """Summarize the following article in a concise and informative paragraph. Focus on key events, important facts, and any significant conclusions.The paragraph should not exceed 100 words."""
prompt_llama = """You are a helpful assistant. Please read the following article and write an abstractive summary. Focus on key events, major takeaways, and eliminate redundancy. The paragraph should not exceed 100 words."""
prompt_gemma = """Summarize the following article in clear, concise and informative paragraph. Highlight the main points, important events, and key conclusions.The paragraph should not exceed 100 words."""

print('-------llama-3.3-70b-versatile----------')
groq_summarize_article(article_text, "llama-3.3-70b-versatile", prompt_llama)

print("-------gemma2-9b-it----------")
groq_summarize_article(article_text, "gemma2-9b-it",prompt_gemma)

print('-------gemini-2.0-flash----------')
gemini_summarize_article(article_text, prompt_gemini)

print("-------------- Article -------------")
print(article_text)