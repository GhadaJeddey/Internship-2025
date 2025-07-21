import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class NERModel:
    def __init__(self, config):
        
        self.spacy_nlp = spacy.load(config['model']['spacy_model'])
        
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(config['model']['roberta_model'])
        self.roberta_model = AutoModelForTokenClassification.from_pretrained(config['model']['roberta_model'])
        
        self.roberta_nlp = pipeline(
            "ner",
            model=self.roberta_model,
            tokenizer=self.roberta_tokenizer,
            aggregation_strategy="simple"
        )
        
        self.entity_mapping = {}

    def predict_spacy(self, text):
        doc = self.spacy_nlp(text)
        return [{
            "entity": ent.label_,
            "word": ent.text,
            "score": 1.0, 
            "start": ent.start_char,
            "end": ent.end_char,
            "mapped_id": self.entity_mapping.get(ent.text.strip(), "N/A")
        } for ent in doc.ents]

    def predict_roberta(self, text):
        results = self.roberta_nlp(text)
        return [{
            "entity": ent["entity_group"],
            "word": ent["word"],
            "score": ent["score"],
            "start": ent["start"],
            "end": ent["end"],
            "mapped_id": self.entity_mapping.get(ent["word"].strip(), "N/A")
        } for ent in results]

    def merge_entities(self, spacy_entities, roberta_entities):
        merged = []
        roberta_types = {"PER", "ORG", "LOC", "MISC"}
        spacy_types = {
            "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW",
            "MONEY", "NORP", "ORDINAL", "PERCENT", "PRODUCT", "QUANTITY",
            "TIME", "WORK_OF_ART"
        }

        for entity in roberta_entities:
            if entity["entity"] in roberta_types:
                merged.append(entity)

        
        for entity in spacy_entities:
            if entity["entity"] in spacy_types:
                overlap = any(
                    not (entity["end"] <= m["start"] or entity["start"] >= m["end"])
                    for m in merged
                )
                if not overlap:
                    merged.append(entity)

        return sorted(merged, key=lambda x: x["start"])

    def predict(self, text):
        spacy_entities = self.predict_spacy(text)
        roberta_entities = self.predict_roberta(text)
        return self.merge_entities(spacy_entities, roberta_entities)
