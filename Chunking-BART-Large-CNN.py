import json
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import nltk
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')
#nltk.download('punkt_tab')  
import json
from langchain.text_splitter import TokenTextSplitter
from transformers import BartTokenizer
from evaluate import load 


""" 
    Functions : 
    - token_counter(): Counts the number of tokens in a given text using the BART tokenizer.
        Chunking Methods:
        - normal_chunking(): Splits the text into smaller chunks using a sliding window approach (overlapping chunks).
        - semantic_chunking(): Splits text into chunks based on semantic boundaries (paragraphs).
    
        Summarization Approaches:
        - progressive_context_aware_summarization(): Summarizes text progressively, using previous summaries as context.
        - semantic_then_progressive(): Combines semantic chunking with progressive summarization.
        - summarize_text(): Summarizes text, handling long texts by chunking and progressive summarization.
        - summarize_long_text(): Handles long texts by chunking and progressive summarization.
        - evaluation(): Evaluates the summary using ROUGE, BLEU, and BERTScore metrics.
    
"""

# Initialize model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def token_counter(tokenizer, article):
    tokens = tokenizer(article, return_tensors="pt", truncation=False)
    num_tokens = tokens.input_ids.shape[1]
    return num_tokens

def normal_chunking(text, chunk_size=400, stride=100):
    
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,        # total tokens per chunk
        chunk_overlap=stride,     # overlap in tokens
        
    )
    chunks = splitter.split_text(sample_text)
    return chunks

def semantic_chunking(text, max_tokens_per_chunk=400): 
    """Simple semantic chunking by paragraph (assuming \n\n separation)"""
    paragraphs = text.split('\n\n')
    chunks = []
    for para in paragraphs:
        tokens = tokenizer.encode(para, add_special_tokens=False)
        if len(tokens) <= max_tokens_per_chunk:
            chunks.append(para)
            
        else:
            # fallback to sliding if paragraph too long
            chunks.extend(normal_chunking(para, chunk_size=max_tokens_per_chunk))
    return chunks


def sliding_window_approach(chunks):
    concat_summaries = ""
    for i, chunk in enumerate(chunks):
        summary = summarize_text(chunk)
        concat_summaries += summary + " "
    summary = summarize_text(concat_summaries)
    print(f"Final summary length: {len(concat_summaries)} chars")
    return summary.strip()

def progressive_approach(chunks, max_length=500, min_length=100, num_beams=2, length_penalty=2.0):
    
    """Safe progressive summarization that handles token limits"""
    prev_summary = " "
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Combine previous summary with current chunk
        if prev_summary:
            combined_text = prev_summary + " " + chunk 
        else:
            combined_text = chunk 
        
        # Check if combined text is still too long
        combined_tokens = len(tokenizer.encode(combined_text, add_special_tokens=False))
        
        if combined_tokens > 1000:  # Leave buffer for special tokens
            # If too long, just use the chunk (lose some context but avoid errors)
            print(f"Combined text too long ({combined_tokens} tokens), using chunk only")
            input_text = chunk
        else:
            input_text = combined_text
        
        prev_summary = summarize_text(input_text, max_length=max_length, min_length=min_length,
                                       num_beams=num_beams, length_penalty=length_penalty) 
    final_summary = prev_summary.strip()
    
    return final_summary

def summarize_text(text, max_length=500, min_length=100, num_beams=2 , length_penalty=2.0,summarization_method='sliding_window_approach',chunking_method='normal_chunking'):
    # Check token count first
    token_count = token_counter(tokenizer, text)
    
    # If text is too long, use chunking approach
    if token_count > 1024:
        print(f"Text too long ({token_count} tokens). Using chunking approach...")
        return summarize_long_text(text, max_length, min_length, num_beams, length_penalty, summarization_method, chunking_method)
    
    # If text is short enough, summarize directly
    inputs = tokenizer.encode(text, 
                              return_tensors="pt", 
                              truncation=True, 
                              max_length=1024
                              ).to(device)
    
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length,
                                 num_beams=num_beams, length_penalty=length_penalty, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_long_text(text, max_length=500, min_length=100, num_beams=2, length_penalty=2.0,summarization_method='sliding_window_approach',chunking_method='normal_chunking'):
    """Handle long texts by chunking and progressive summarization"""
    
    if chunking_method == 'semantic_chunking':
        chunks = semantic_chunking(text)
    else : 
        chunks = normal_chunking(text)
    
    if summarization_method == 'sliding_window_approach':
        return sliding_window_approach(chunks)

    if summarization_method == 'progressive_approach':
        return progressive_approach(chunks)


def evaluation(article, summary): 
    rouge = load("rouge")
    bleu = load("bleu")
    bertscore = load("bertscore")
    
    rouge_score = rouge.compute(predictions=[summary], references=[article])
    bleu_score = bleu.compute(predictions=[summary], references=[[article]])
    bert_score = bertscore.compute(predictions=[summary], references=[article],lang="en")

    return rouge_score, bleu_score, bert_score

if __name__ == "__main__":
    
    from colorama import Fore, Style, init
    init(autoreset=True)  # Automatically reset colors after each print

    # Load long article
    with open('novatech_article.txt', 'r', encoding='utf-8') as f:
        sample_text = f.read()
    

    # Number of tokens
    num_tokens = token_counter(tokenizer, sample_text)
    print(f"\n{Fore.YELLOW} Total Tokens in the Article: {Fore.CYAN}{num_tokens}")


    # --- Test Approach ---
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.BLUE} Starting: Semantic Chunking + Sliding Window (summarized concatenated summaries)")
    print(f"{Fore.MAGENTA}{'='*60}\n")

    chunks = semantic_chunking(sample_text)
    summary = progressive_approach(chunks)
    
    print(f"{Fore.BLUE} Final Summary:\n{Fore.RESET}{summary}\n")
    print(f"{'='*60}")
    print(f"{Fore.BLUE} Evaluation Results:")
    print(f"{'='*60}\n")
    rouge_score, bleu_score, bert_score = evaluation(sample_text, summary)
    print(f"ROUGE Score: {rouge_score}")
    print(f"BLEU Score: {bleu_score}")
    print(f"BERTScore: {bert_score}")
    print(f"{'='*60}\n")
    