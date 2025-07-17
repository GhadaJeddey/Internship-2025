from transformers import pipeline, AutoTokenizer
import ast 
import pandas as pd
import spacy 

class NERmodel: 
    def __init__(self, model_name,file_path=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=model_name, grouped_entities=True, tokenizer=self.tokenizer)
        self.spacy_nlp = spacy.load("en_core_web_trf")
        self.df = pd.read_csv(file_path) if file_path else None
    
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
    
    
    def sentence_extractor(self, sentence_id, column_name="Word"):
        """
        Extracts a sentence based on the sentence ID and column name.
        """
        sentence = self.df.loc[self.df['Sentence_ID'] == "Sentence: " + str(sentence_id), column_name].values[0]
        res = ""
        for s in sentence.split(','):
            res = res + " " + s.strip()[1:-1]
        return res.strip()[1:-1]
    
    def compare_tags(self, expected_tags, ner_tags):
        """
        Compare NER results with expected tags from the dataset.
        """
        if expected_tags == ner_tags:
            print("NER results match expected tags.")
        else:   
            print("NER results do not match expected tags.")
            print(f"Expected: {expected_tags}")
            print(f"Got: {ner_tags}")

    def run_simple_ner(self, sentence):
        
        return self.ner_pipeline(sentence)
    
    def run_hybrid_ner(self, sentence):
  
        """
        Run both transformer-based and spaCy-based NER, combining results.
        SpaCy focuses on DATE, TIME, MONEY, etc., while transformer handles standard entities.
        """
        hf_ents = self.ner_pipeline(sentence)

        doc = self.spacy_nlp(sentence)
        spacy_ents = [
            {
                "entity_group": ent.label_,
                "word": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "score": "spaCy"
            }
            for ent in doc.ents
            if ent.label_ in {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL","GPE","NORP"}
        ]

        return sorted(hf_ents + spacy_ents, key=lambda x: x["start"])
    
    
if __name__ == "__main__":
    
    model_name = "Jean-Baptiste/roberta-large-ner-english"
    ner = NERmodel(model_name=model_name)
    ner.load_data("NER_Dataset.csv")
    sentence_id = 5
    sentence = ner.sentence_extractor(sentence_id)
    #sentence = "Pour la livraison urgente de notre nouveau modèle d'ordinateur portable, nous avons contacté le PDG de l'entreprise, espérant qu'il pourrait accélérer le processus via le service de messagerie express, et ainsi respecter la DPA que nous avons signée avec le client, tout en minimisant notre ROI et maximisant la satisfaction du client, sans oublier la TVA qui sera appliquée."
    
    """import json
    with open('articles.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    sentence = articles['reference'] """ 
    print(f"Sentence : {sentence}")
    
    hybrid_results = ner.run_hybrid_ner(sentence)
    print("----------NER Results------------")
    for entity in hybrid_results:
        print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Start: {entity['start']}, End: {entity['end']}, Score: {entity['score']}")
    
    print("\n Expected Tags:")
    expected = ner.sentence_extractor(sentence_id, column_name="Tag")
    print(expected)
    
