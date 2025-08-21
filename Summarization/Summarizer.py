from random import sample
import torch, json, time
from transformers import GenerationConfig 
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.text_splitter import TokenTextSplitter



class Summarizer : 
    def __init__(self, model_name="facebook/bart-large-cnn"): 

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_bart_base_cnn(self,input_text):
        """ Test BART-base-cnn model """ 
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        generation_config = GenerationConfig(
            input_ids=input_ids,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            length_penalty=2.0,
            max_length=128,
            min_length=56,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        summary_ids = self.model.generate(
            input_ids,
            generation_config=generation_config
        )
        print(self.tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    def token_counter(self, article):
        tokens = self.tokenizer(article, return_tensors="pt", truncation=False)
        num_tokens = tokens.input_ids.shape[1]
        return num_tokens

    def chunking_token_based(self,text, chunk_size=400, stride=100):
        """
        Token-based chunking with overlap between two consecutive chunks 
        Args:
            text (str): The input text to be chunked.
            chunk_size (int): Total tokens per chunk.
            stride (int): Overlap in tokens between chunks.
        Returns:
           list: A list of text chunks.
        """
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,        
            chunk_overlap=stride,  
        )
        
        chunks = splitter.split_text(text)
        return chunks

    def chunking_paragraph_based(self,text, max_tokens_per_chunk=400): 
    
        """Simple semantic chunking by paragraph (assuming \n\n separation) no sliding window """
        
        paragraphs = text.split('\n\n')
        chunks = []
        for para in paragraphs:
            tokens = self.tokenizer.encode(para, add_special_tokens=False)
            if len(tokens) <= max_tokens_per_chunk:
                chunks.append(para)
                
            else:
                # fallback if paragraph too long
                chunks.extend(self.chunking_token_based(para, chunk_size=max_tokens_per_chunk))
        return chunks


    def progressive_summary(self,chunks, max_length=500, min_length=100, num_beams=2, length_penalty=2.0):
    
        """Safe progressive summarization that handles token limits : Combine previous summary with current chunk
            Args : 
                chunks (list): List of text chunks to summarize.
                max_length (int): Maximum length of the summary.
                min_length (int): Minimum length of the summary.
                num_beams (int): Number of beams for beam search.
                length_penalty (float): Length penalty for the summary.
            Returns : 
                str: The final summary after processing all chunks.
        """
        
        prev_summary = " "
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")

            if prev_summary:
                combined_text = prev_summary + " " + chunk 
            else:
                combined_text = chunk 

            tokens_length = len(self.tokenizer.encode(combined_text, add_special_tokens=False))

            if tokens_length > 1000: 
                # If too long, use the chunk without the previous summary 
                print(f"Combined text too long ({tokens_length} tokens), using chunk only ..")
                input_text = chunk
            else:
                input_text = combined_text
            
            prev_summary = self.summarize_text(input_text, max_length=max_length, min_length=min_length,
                                        num_beams=num_beams, length_penalty=length_penalty) 
        final_summary = prev_summary.strip()
        
        return final_summary
    
    def summarize_text(self,text, max_length=500, min_length=100, num_beams=2 , length_penalty=2.0):
       
        token_count = self.token_counter(text)
        
        if token_count > 1024:
            print(f"Text too long ({token_count} tokens). Using chunking approach...")
            chunks = self.chunking_paragraph_based(text)
            return self.progressive_summary(chunks)
        
        # If text is short enough, summarize directly 
        inputs = self.tokenizer.encode(text, 
                                return_tensors="pt", 
                                truncation=True, 
                                max_length=1024
                                ).to(self.device)
        
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length,
                                    num_beams=num_beams, length_penalty=length_penalty, early_stopping=True)

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  
    def evaluation(self,reference_summary, summary): 
        rouge = load("rouge")
        bleu = load("bleu")
        bertscore = load("bertscore")

        rouge_score = rouge.compute(predictions=[summary], references=[reference_summary])
        bert_score = bertscore.compute(predictions=[summary], references=[reference_summary],lang="en")

        return rouge_score, bert_score
    
    def evaluate_on_all_articles(self, md_path="articles-gemini.md", output_path="summarization-results-compared-to-gemini.json"):
        with open(md_path, "r", encoding="utf-8") as f:
            samples = f.read().split("# Article")[1:]

        all_rouge = []
        all_bert = []
        results = []

        for idx, sample in enumerate(samples):
            article = sample.split("# Reference")[0]
            reference = sample.split("# Reference")[1]
            summary = self.summarize_text(article)
            rouge_score, bert_score = self.evaluation(reference, summary)

            rouge_metrics = {}
            for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                if key in rouge_score:
                    rouge_metrics[key] = rouge_score[key]

            rouge_f1 = rouge_score["rougeL"] if "rougeL" in rouge_score else 0
            bert_f1 = sum(bert_score["f1"]) / len(bert_score["f1"]) if bert_score["f1"] else 0
            all_rouge.append(rouge_f1)
            all_bert.append(bert_f1)
            print(f"Article {idx+1}: ROUGE-L F1={rouge_f1:.4f}, BERTScore F1={bert_f1:.4f}")
            results.append({
                "article_index": idx+1,
                "reference_summary": reference.strip(),
                "generated_summary": summary,
                "rouge_metrics": rouge_metrics,
                "rougeL_f1": rouge_f1,
                "bert_f1": bert_f1
            })

        avg_rouge = sum(all_rouge) / len(all_rouge) if all_rouge else 0
        avg_bert = sum(all_bert) / len(all_bert) if all_bert else 0
        print(f"\nAverage ROUGE-L F1: {avg_rouge:.4f}")
        print(f"Average BERTScore F1: {avg_bert:.4f}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "average_rougeL_f1": avg_rouge,
                "average_bert_f1": avg_bert
            }, f, indent=2)
        return avg_rouge, avg_bert
    
if __name__ == "__main__":

    summarizer = Summarizer()
    summarizer.evaluate_on_all_articles()
