from random import sample
import torch, json, time
from transformers import GenerationConfig 
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.text_splitter import TokenTextSplitter


'''
    Functions : 
    - token_counter : return the number of tokens in the input text 
    - chunking_token_based : 
    - chunking_paragraph_based
    - progressive_summary
    - summarize_text

    Logic : 
    - If text is short (less than 1024 token) , summarize directly 
    - If text is long (more than 1024 tokens), use chunking : 
        -- The chunking is paragraph based by default 
        -- If the paragraph itself is still long , or no paragraphs are to be found in the article , use token-based chunking with overlap between chunks 
    - The summary of a previous chunk is concatenated with the next chunk and fed into the model to summarize again (process repeated until the entire text is summarized)

'''

class Summarizer : 
    def __init__(self, model_name: str ="facebook/bart-large-cnn" ) -> None: 
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def token_counter(self, article :str) -> int:
        ''' 
        Count the number of tokens in the input article.
        Args :
            article (str): The input article text.
        Returns : 
            int: The number of tokens in the input article.
        '''
        tokens = self.tokenizer(article, return_tensors="pt", truncation=False)
        num_tokens = tokens.input_ids.shape[1]
        return num_tokens

    def chunking_token_based(self,text: str, chunk_size:int =400, stride:int =100) -> list:
        ''''
        Token-based chunking with overlap between two consecutive chunks 
        Args:
            text (str): The input text to be chunked.
            chunk_size (int): Total tokens per chunk.
            stride (int): Overlap in tokens between chunks.
        Returns:
           list: A list of text chunks.
           
        '''
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,        
            chunk_overlap=stride,  
        )
        
        chunks = splitter.split_text(text)
        return chunks

    def chunking_paragraph_based(self,text :str, max_tokens_per_chunk :int =400) -> list:

        '''
        Simple semantic chunking by paragraph (assuming \n\n separation) no sliding window 
        Args :
            text (str): The input text to be chunked.
            max_tokens_per_chunk (int): Maximum tokens allowed per chunk.
        Returns : 
            list: A list of text chunks.
        '''
        
        paragraphs = text.split('\n\n')
        chunks = []
        for para in paragraphs:
            tokens = self.tokenizer.encode(para, add_special_tokens=False)
            if len(tokens) <= max_tokens_per_chunk:
                chunks.append(para)
                
            else:
                # fallback if paragraph itself is too long or no paragraphs are found 
                chunks.extend(self.chunking_token_based(para, chunk_size=max_tokens_per_chunk))
        return chunks


    def progressive_summary(self,chunks: list, max_length:int =500, min_length :int=100, num_beams:int =2, length_penalty:int =2.0) -> str:

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

            if prev_summary:
                combined_text = prev_summary + " " + chunk 
            else:
                combined_text = chunk 

            tokens_length = len(self.tokenizer.encode(combined_text, add_special_tokens=False))

            if tokens_length > 1000: 
                # If too long, use the chunk without the previous summary 
                input_text = chunk
            else:
                input_text = combined_text
            
            prev_summary = self.summarize_text(input_text, max_length=max_length, min_length=min_length,
                                        num_beams=num_beams, length_penalty=length_penalty) 
        final_summary = prev_summary.strip()
        
        return final_summary

    def summarize_text(self, text: str, max_length: int = 500, min_length: int = 100, num_beams: int = 2, length_penalty: float = 2.0) -> str:
        '''
        Summarize the input text using the model.
        Args:
            text (str): The input text to be summarized.
            max_length (int): Maximum length of the summary.
            min_length (int): Minimum length of the summary.
            num_beams (int): Number of beams for beam search.
            length_penalty (float): Length penalty for the summary.
        Returns:
            str: The generated summary.
        
        '''
        token_count = self.token_counter(text)

        # if text is long -> use chunking 
        if token_count > 1024:
            chunks = self.chunking_paragraph_based(text)
            return self.progressive_summary(chunks)
        
        # If text is short enough -> summarize directly 
        inputs = self.tokenizer.encode(text, 
                                return_tensors="pt", 
                                truncation=True, 
                                max_length=1024
                                ).to(self.device)
        
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length,
                                    num_beams=num_beams, length_penalty=length_penalty, early_stopping=True)

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def evaluation(self, reference_summary: str, summary: str) -> tuple:
        '''
            Evaluate the generated summary against the reference summary using various metrics.
            Args :
                reference_summary (str): The reference summary to compare against.
                summary (str): The generated summary to evaluate.
            Returns : 
                rouge_score (dict): The ROUGE score metrics.
                bert_score (int): The BERT score metrics.
        '''
        rouge = load("rouge")
        bleu = load("bleu")
        bertscore = load("bertscore")

        rouge_score = rouge.compute(predictions=[summary], references=[reference_summary])
        bert_score = bertscore.compute(predictions=[summary], references=[reference_summary],lang="en")

        return rouge_score, bert_score

    def evaluate_on_all_articles(self, md_path: str = "Summarization-task/articles.md", output_path: str = "Summarization-task/summarization-final-results.json") -> tuple:
        '''
        Evaluate the summarization model on all articles.   
        Args : 
            md_path (str): The path to the markdown file containing articles.
            output_path (str): The path to the output JSON file for saving results.
        Returns : 
            tuple: A tuple containing the average ROUGE-1, ROUGE-2, ROUGE-L, and BERT scores.
        '''
        
        with open(md_path, "r", encoding="utf-8") as f:
            samples = f.read().split("# Article")[1:]

        all_rouge1 = []
        all_rouge2 = []
        all_rougeL = []
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

            rouge1_f1 = rouge_score["rouge1"] if "rouge1" in rouge_score else 0
            rouge2_f1 = rouge_score["rouge2"] if "rouge2" in rouge_score else 0
            rougeL_f1 = rouge_score["rougeL"] if "rougeL" in rouge_score else 0
            bert_f1 = sum(bert_score["f1"]) / len(bert_score["f1"]) if bert_score["f1"] else 0
            all_rouge1.append(rouge1_f1)
            all_rouge2.append(rouge2_f1)
            all_rougeL.append(rougeL_f1)
            all_bert.append(bert_f1)
            print(f"Article {idx+1}: ROUGE-1 F1={rouge1_f1:.4f}, ROUGE-2 F1={rouge2_f1:.4f}, ROUGE-L F1={rougeL_f1:.4f}, BERTScore F1={bert_f1:.4f}")
            results.append({
                "article_index": idx+1,
                "reference_summary": reference.strip(),
                "generated_summary": summary,
                "rouge_metrics": rouge_metrics,
                "rouge1_f1": rouge1_f1,
                "rouge2_f1": rouge2_f1,
                "rougeL_f1": rougeL_f1,
                "bert_f1": bert_f1
            })

        avg_rouge1 = sum(all_rouge1) / len(all_rouge1) if all_rouge1 else 0
        avg_rouge2 = sum(all_rouge2) / len(all_rouge2) if all_rouge2 else 0
        avg_rougeL = sum(all_rougeL) / len(all_rougeL) if all_rougeL else 0
        avg_bert = sum(all_bert) / len(all_bert) if all_bert else 0
        print(f"\nAverage ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-2 F1: {avg_rouge2:.4f}")
        print(f"Average ROUGE-L F1: {avg_rougeL:.4f}")
        print(f"Average BERTScore F1: {avg_bert:.4f}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "average_rouge1_f1": avg_rouge1,
                "average_rouge2_f1": avg_rouge2,
                "average_rougeL_f1": avg_rougeL,
                "average_bert_f1": avg_bert
            }, f, indent=2)
        return avg_rouge1, avg_rouge2, avg_rougeL, avg_bert
    
if __name__ == "__main__":

    summarizer = Summarizer()
    summarizer.evaluate_on_all_articles()
    
