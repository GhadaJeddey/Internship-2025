from transformers import BartTokenizer
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from transformers import GenerationConfig 
from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json 
from evaluate import load
import time 

""" Test BART models :
    - BART-base-cnn
    - BART-large
    - BART-large-cnn

"""

def token_counter(tokenizer, article):
    tokens = tokenizer(article, return_tensors="pt", truncation=True)
    num_tokens = tokens.input_ids.shape[1]
    print(f"Number of tokens: {num_tokens}")


def formatted_output(text, split_criteria='.'):
    
    for line in str(text).split(split_criteria):
        if line.strip():
            print(line.strip())

def evaluation(summary , article ) :
      # Initialize metrics
    rouge = load("rouge")
    bertscore = load("bertscore")
   
    # Compute ROUGE scores
    rouge_results = rouge.compute(
        predictions=[summary],
        references=[article],
        use_stemmer=True
    )
    
    # Compute BERTScore
    bert_results = bertscore.compute(
        predictions=[summary],
        references=[article],
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli"  # Best model for BERTScore
    )
    
    
    
    # Print formatted results
    print("\n--- ROUGE RESULTS ---")
    print(f"ROUGE-1: {rouge_results['rouge1']:.3f}")
    print(f"ROUGE-2: {rouge_results['rouge2']:.3f}")
    print(f"ROUGE-L: {rouge_results['rougeL']:.3f}")
    
    print("\n--- BERTSCORE RESULTS ---")
    print(f"Precision: {bert_results['precision'][0]:.3f}")
    print(f"Recall: {bert_results['recall'][0]:.3f}")
    print(f"F1: {bert_results['f1'][0]:.3f}")
    

    return {
        "rouge": rouge_results,
        "bertscore": bert_results,
      

    }
  
def test_bart_base(input_text):
    """ Test BART-base model for text generation (e.g. summarization) """
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs= inputs["input_ids"],
                                 max_length=128, 
                                 num_beams=4, #test 5 then 6 
                                 early_stopping=False,
                                 do_sample=False,
                                 )
    
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(output_text)


def test_bart_base_cnn(input_text):
    """ Test BART-base-cnn model """ 
    tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    model = BartForConditionalGeneration.from_pretrained("ainize/bart-base-cnn")
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    

    generation_config = GenerationConfig(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=2.0,
        max_length=128,
        min_length=56,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    summary_ids = model.generate(
        input_ids,
        generation_config=generation_config
    )
    print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    

def test_bart_large(input_text):
    """ Test BART-large model"""
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs= inputs["input_ids"],
                                 max_length=128, 
                                 num_beams=4, #test 5 then 6 
                                 early_stopping=False,
                                 do_sample=False,
                                 )
    output = test_t5_large(article)
    evaluation(output, article)
  
    
def test_bart_large_cnn(input_text):
    """ Test BART-large-cnn model """
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    token_counter(tokenizer, input_text)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    
    inputs = tokenizer.encode(input_text,
                              max_length=1024,
                              truncation=True,
                              return_tensors="pt")
    
    start_time = time.time()
    summary_ids = model.generate(
        inputs,
        max_length=256,
        min_length=50,
        num_beams=2,
        length_penalty=2.0 ,
        
        early_stopping=True,
        
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    end_time = time.time()
    print(f"Time taken for summarization: {end_time - start_time:.2f} seconds")
    print("----------- Summary -----------")
    print( summary)
    return summary

def test_t5_large(input_text):
    """ Test T5-large model """
    
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5ForConditionalGeneration.from_pretrained('t5-large')

    inputs = tokenizer("summarize: "+input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(inputs["input_ids"],
                                 max_length=128,
                                 num_beams=4,
                                 early_stopping=True,
                                 length_penalty=1.5,
                                 no_repeat_ngram_size=3,
                                 repetition_penalty=1.2,
                                 )
    
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(output_text)
    return output_text

def test_pegasus_cnn_dailymail(input_text):
    """ Test Pegasus model for CNN/DailyMail summarization """
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")

    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(inputs["input_ids"],
                                 max_length=128,
                                 num_beams=4,
                                 early_stopping=True,
                                 length_penalty=1.5,
                                 no_repeat_ngram_size=3,
                                 repetition_penalty=1.2,
                                 )
    
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(output_text)
    return output_text

def pegasus_large(input_text):
    """ Test Pegasus-large model """
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large")

    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(inputs["input_ids"],
                                 max_length=128,
                                 num_beams=3,
                                 early_stopping=True,
                                 
                                 )
    
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(output_text)
    return output_text
    
if __name__ == "__main__":
    
    input_text= """You’ve probably heard the old canard that new brain cells simply stop forming as we become adults. But research out today is the latest to show that this isn’t really true.
    Scientists in Sweden led the study, published Thursday in Science. They found abundant signs of neural stem cells growing in the hippocampus of adult brains. The findings reveal more about the human brain as we get older, the researchers say, and also hint at potential new ways to treat neurological disorders.
    “We’ve found clear evidence that the human brain keeps making new nerve cells well into adulthood,” study co-author Marta Paterlini, a neuroscientist at the Karolinska Institute in Stockholm, told Gizmodo.
    This isn’t the first paper to chip away at the idea of new neurons ceasing to form in adulthood (a concept not to be confused with general brain development, which does seem to reach maturity around age 30). In 2013, study researcher Jonas Frisén and his team at the Karolinska Institute concluded that substantial neuron growth—also known as neurogenesis—occurs throughout our lives, albeit with a slight decline as we become elderly. But there’s still some debate ongoing among scientists. In spring 2018, for instance, two different studies of neurogenesis published a month apart came to the exact opposite conclusion.
    The researchers were hoping to settle one particular aspect of human neurogenesis in adults. If we do keep growing new neurons as we age, then we should be able to spot the cells that eventually mature into neurons, neural progenitor cells, growing and dividing inside the adult brain. To look for these cells, the team analyzed brain tissue samples from people between the ages of 0 and 78 using relatively new advanced methods. These methods allowed them to figure out the characteristics of brain cells on an individual level and to track the genes being expressed by a single cell’s nucleus.
    All told, the researchers examined more than 400,000 individual cell nuclei from these samples. And as hoped, they found these progenitor cells along various stages of development in adult brains, including cells just about to divide. They also pinpointed the location within the hippocampus where the new cells appeared to originate: the dentate gyrus, a brain region critical to helping us form certain types of memory.
    “We saw groups of dividing precursors sitting right next to the fully formed nerve cells, in the same spots where animal studies have shown adult stem cells live,” said Paterlini, a senior scientist at the Frisén lab. “In short, our work puts to rest the long-standing debate about whether adult human brains can grow new neurons.”
    The findings, as is often true in science, foster more questions in need of an answer.Our adult precursor cells seem to have different patterns of gene activity compared to the cells found in pigs, mice, and other mammals with clear evidence of adult neurogenesis, for instance.
    The researchers also found that some adults’ brains were filled with these growing precursors, while others had relatively few. These differences—combined with the team’s earlier research showing that adult neurogenesis slows down over time—may help explain people’s varying risk of neurological or psychological conditions, the authors say. And likewise, finding a safe way to improve the adult brain’s existing ability to grow new cells could help treat these conditions or improve people’s recovery from serious head injuries.
    “Although precise therapeutic strategies for humans are still being researched, the simple fact that our adult brains can generate new neurons radically changes the way we view lifelong learning, recovery from injury, and the untapped potential of neuronal plasticity,” said Paterlini.
    There’s plenty more to be learned about how our brains change over time. The team is planning to investigate other likely hotspots of neurogenesis in the adult brain, such as the wall of the lateral ventricles (c-shaped cavities found in each of the brain’s cerebral hemispheres) and nearby regions. But we can be fairly certain that our neurons keep on growing and replacing themselves into adulthood—at least for some of us."""    
    
   
    with open('articles.json', 'r', encoding='utf-8') as f:
        articles = json.load(f) 
        
    article_key = "Medium-1"  
    article= articles.get(article_key) 
    print("\n\n")
    #input_text = articles.get("Medium-2")
     
    #article = article * 2
    #article = article +"\n" +input_text
    summarization_bart = test_bart_large_cnn(input_text)

    
    

    
    
    
    
