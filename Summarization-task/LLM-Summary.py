import json 
import requests 
import os 
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from groq import Groq


load_dotenv('.env')

GROK_API_KEY = os.environ.get("GROK_API_KEY") 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

def groq_summarize_article(article_text,model,prompt):
    client = Groq(
        api_key=GROK_API_KEY,
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
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    response = model.generate_content(
        str(prompt) + f" .Article : {article_text}",
        generation_config=GenerationConfig(max_output_tokens=500)
    )

    summary = response.text
    print("Summary:", summary)
    return summary


article_text =""" The corporate landscape is facing a barrage of negative headlines as new cases of financial misconduct and cybercrime emerge globally. In the UK, the Serious Fraud Office (SFO) has been particularly active, charging six individuals in a complex £75 million pension fraud case and freezing over £10,000 in cryptocurrency assets belonging to the CEO of Arena TV. These actions are part of a broader crackdown on sophisticated financial crimes, which has also seen a former Goldman Sachs analyst ordered to pay over half a million pounds for an insider trading conviction. In the U.S., a massive health care fraud takedown has uncovered schemes worth over a billion dollars in false claims, with numerous individuals facing charges for billing Medicare for unnecessary services.

Adding to the turmoil, major companies and government agencies are battling a relentless onslaught of cyberattacks and data breaches. A recent report from IT Governance revealed that 14.9 million records were breached in July alone, with major incidents at companies like Allianz Life and Qantas. The breaches underscore a growing trend of third-party vendor compromises, where a security flaw in a smaller company's system can expose data from a much larger partner. This trend has made the supply chain a primary target for hackers.  In a particularly alarming case, a cyberattack on Iran's Bank Sepah resulted in the theft of 42 million customer records, and the U.S. government's TeleMessage app, used by officials, was breached, exposing metadata from over 60 accounts. These incidents highlight the increasing sophistication of criminal and state-sponsored hacking groups who are now targeting critical infrastructure and sensitive government communications.
"""
prompt_gemini = """Summarize the following article in a concise and informative paragraph. Focus on key events, important facts, and any significant conclusions.The paragraph should not exceed 100 words."""

gemini_summarize_article(article_text, prompt_gemini)

