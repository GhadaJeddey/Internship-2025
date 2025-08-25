
import json
import re
from typing import Dict, List, Tuple, Set
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

"""
Class RelationExtractor : 
  Functions : 
  - create_prompt : prompt based on text length 
  - extract_relations : Runs the full extraction pipeline: prepares the prompt, gets the model’s response, parses and validates the extracted relations, and returns the results.
  - validate_relations : Checks that each extracted relation is well-formed, uses supported relation types, and refers only to entities present in the input.

"""

class RelationExtractor:
    def __init__(self, quantized_8 = False ):
        """Initialize a completely generic relation extraction system."""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained( "Qwen/Qwen2.5-7B-Instruct",trust_remote_code=True)

        self.initialize_model(quantized_8)
        self.initialize_schema()

    def initialize_model(self,quantized_8):
        """Load the model with safe defaults for any text."""

        if quantized_8: 
            try : 
              self.model = AutoModelForCausalLM.from_pretrained(
                  "Qwen/Qwen2.5-7B-Instruct",
                  device_map="auto",
                  load_in_8bit=True,
                  trust_remote_code=True
              )
            except Exception as e:
              raise RuntimeError(f"Model initialization failed: {str(e)}")

        else :
          try :
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                device_map="auto",
                trust_remote_code=True
            )
          except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")          

        self.generation_config = {
                  "max_new_tokens": 2048,
                  "temperature": 0.1,
                  "do_sample": True,
                  "top_p": 0.95,
                  "repetition_penalty": 1.05,
                  "pad_token_id": self.tokenizer.eos_token_id
              }

    def initialize_schema(self):

        """Define a comprehensive relation schema with good descriptions."""

        self.relation_schema = {
           "person_relations": {

                "per:works": "Person has a  job title or position",
                "per:education": "Person attended or graduated from an educational institution",
                "per:family": "Family relationship between two people (parent, sibling, spouse, etc.)",
                "per:born_in": "Person was born in a specific location",
                "per:lived_in": "Person lived or resides in a location",

            },
            "organization_relations": {
                "org:founded_by": "Organization was founded or established by a person or entity",
                "org:founded_in": "Date of founding",

                "org:subsidiaries": "Organization owns or controls another organization as a subsidiary",
                "org:competitors": "Organizations that compete in the same market or industry",
                "org:partners": "Organizations that have business partnerships or alliances",
                "org:products": "Products, services, or works created by an organization"


            },
            "location_relations": {
                "geo:located_in": "location of an entity",
                "geo:capital_of": "City serves as the capital of a country or region",

            },

            "financial_relations": {
                "fin:worth": "Entity has a specific monetary value or net worth",
                "fin:costs": "Something costs a specific amount of money",
                "fin:funded_by": "Project or entity received funding from another entity"
            },

        }

        self.all_relations = [rel for category in self.relation_schema.values() for rel in category.keys()]
        self.supported_entity_types = {"PERSON", "ORG", "LOC", "FAC", "DATE", "TIME", "MONEY", "PERCENT", "LAW", "CARDINAL", "ORDINAL", "WORK_OF_ART", "NORP"}



    def create_prompt(self, text: str, entities: Dict[str, List[str]]) -> str:

        """Create a completely generic prompt without dataset-specific examples."""

        entity_list = []
        for etype, ents in entities.items():
            if etype in self.supported_entity_types:
                for ent in ents:
                    if ent.strip():
                        entity_list.append(f"- {ent} ({etype})")

        schema_desc = []
        for category, relations in self.relation_schema.items():
            schema_desc.append(f"\n{category.replace('_', ' ').title()}:")
            for rel, description in relations.items():
                schema_desc.append(f"  {rel}: {description}")


        text_length = len(text)
        if text_length < 500:
            length_instruction = "This is a short text. Focus on finding all explicit relationships."
        elif text_length < 2000:
            length_instruction = "This is a medium-length text. Look for both direct and contextual relationships."
        else:
            length_instruction = "This is a long text. Thoroughly extract all relationships while maintaining accuracy."

        return f"""Analyze this text and extract all  relationships between entities:

Text: "{text}"

Available Entities:
{chr(10).join(entity_list)}

Possible Relationship Types:
{chr(10).join(schema_desc)}

Instructions:
{length_instruction}
1. Extract only relations explicitly stated in the text
2. Use only the provided entities and relationship types
3. Maintain exact entity names from the provided list
4. Carefully analyze the entire text to find all valid relationships
5. Output ONLY valid JSON with this exact structure:

{{
  "relations": [
    {{
      "subject": "entity_name",
      "relation": "relation_type",
      "object": "entity_name"
    }}
  ]
}}"""

    def generate_response(self, prompt: str) -> str:

        """Generate model response with error handling."""

        try:
            messages = [{"role": "user", "content": prompt}]
            chat_template = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                chat_template,
                return_tensors="pt",
                truncation=False
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )

            return self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            ).strip()

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return ""


    def extract_relations(self, input_data: dict) -> Dict:
        """
        Accepts a dict with keys 'text' and 'entities',
        runs the extraction pipeline, and returns the relations.
        """
        text = input_data.get("text", "")
        entities = input_data.get("entities", {})

        processed_entities = {
            k: [e for e in v if e.strip()]
            for k, v in entities.items()
            if k in self.supported_entity_types
        }

        text_length = len(text)
        prompt = self.create_prompt(text, processed_entities)
        response = self.generate_response(prompt)
        relations = self.parse_response(response).get("relations", [])

        validated_relations = self.validate_relations(relations, processed_entities)

        return {
            "relations": validated_relations
        }

    def parse_response(self, response: str) -> Dict:
        """Robust JSON parsing from model response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            json_str = self.extract_json_string(response)
            try:
                return json.loads(json_str) if json_str else {"relations": []}
            except json.JSONDecodeError:
                print(f"Failed to parse response: {response[:200]}...")
                return {"relations": []}

    def extract_json_string(self, text: str) -> str:

        """Extract JSON string from potentially malformed response."""

        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        if "```json" in text:
            json_part = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_part = text.split("```")[1].split("```")[0].strip()
        else:
            json_part = text

        json_match = re.search(r'\{.*\}', json_part, re.DOTALL)
        return json_match.group(0) if json_match else ""

    def validate_relations(self, relations: List[Dict], entities: Dict[str, List[str]]) -> List[Dict]:

        """Validate extracted relations."""
        valid_relations = []


        entity_lookup = set()
        for etype, ents in entities.items():
            for ent in ents:
                entity_lookup.add(ent.lower())

        for rel in relations:
            if self.is_valid_relation(rel, entity_lookup):

                clean_rel = {
                    "subject": rel["subject"],
                    "relation": rel["relation"],
                    "object": rel["object"]
                }
                valid_relations.append(clean_rel)

        return valid_relations

    def is_valid_relation(self, rel: Dict, entity_lookup: Set[str]) -> bool:

        """Check if a single relation is valid."""

        required_keys = {"subject", "relation", "object"}
        if not all(key in rel for key in required_keys):
            return False


        if rel["relation"] not in self.all_relations:
            return False


        subj = rel["subject"].lower()
        obj = rel["object"].lower()

        if subj not in entity_lookup or obj not in entity_lookup:
            return False

        return True

    def display_relations(self, relations: List[Dict]):

        """Pretty-print relations."""

        if not relations:
            print("No valid relations found")
            return

        print(f"\nFound {len(relations)} valid relations:")
        print("-" * 80)
        print(f"{'Subject':<25} {'Relation':<25} {'Object':<25}")
        print("-" * 80)
        for rel in relations:
            print(
                f"{rel['subject'][:24]:<25} "
                f"{rel['relation'][:24]:<25} "
                f"{rel['object'][:24]:<25}"
            )
        print("-" * 80)

    def process_ner_output(self, ner_data: Dict) -> Dict[str, List[str]]:

        """Process NER output to handle all entity types."""

        normalized = {}
        for etype, ents in ner_data.items():

            clean_ents = [e.strip() for e in ents if e.strip()]
            if clean_ents:
                normalized[etype.upper()] = clean_ents
        return normalized
    
    def re_predict(self, ner_data: dict, text: str):

        """Predict relations between NER entities in the given text."""
        
        ner_result = json.loads(ner_data) if isinstance(ner_data, str) else ner_data
        processed_entities = self.process_ner_output(ner_result)
        result = self.extract_relations({"text": text, "entities": processed_entities})
        return result["relations"]


def run_pipeline():
    """End-to-end pipeline for text input files."""
    print("Starting generic relation extraction pipeline...")
    print("Using Qwen/Qwen2.5-7B-Instruct .")

    try:
        extractor = RelationExtractor()
    except Exception as e:
        print(f"Failed to initialize extractor: {e}")
        return

    try:
        print("\nUpload NER results (JSON):")
        uploaded_files = files.upload()
        if not uploaded_files:
            print("No NER file uploaded.")
            return

        ner_results = json.loads(list(uploaded_files.values())[0].decode('utf-8'))

        print("Upload text file (JSON):")
        uploaded_files = files.upload()
        if not uploaded_files:
            print("No text file uploaded.")
            return

        text_data = json.loads(list(uploaded_files.values())[0].decode('utf-8'))

        results = []


        if isinstance(text_data, dict) and "text" in text_data:

            texts = [text_data["text"]]
        elif isinstance(text_data, list):

            texts = [doc.get("text", doc) if isinstance(doc, dict) else str(doc) for doc in text_data]
        elif isinstance(text_data, str):

            texts = [text_data]
        else:
            print("Invalid text data format. Expected JSON with 'text' field or array of texts.")
            return

        for i, text in enumerate(texts):
            if not text.strip():
                continue

            text_length = len(text)
            print(f"\nProcessing document {i+1}/{len(texts)} ({text_length} characters)...")

            result = extractor.extract_relations(
                {"text": text,
                 "entities": extractor.process_ner_output(ner_results)}
            )

            extractor.display_relations(result["relations"])
            results.append(result) 	

        output_data = {
            "results": results,
            "processing_info": {
                "total_documents": len(results),
                "text_lengths": [r["text_length"] for r in results],
                "total_relations": sum(len(r["relations"]) for r in results)
            }
        }

        output_file = "relation_extraction_results.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        files.download(output_file)
        print(f"\nPipeline completed successfully! Results saved to {output_file}")
        print(f"Processed {len(results)} documents with {sum(len(r['relations']) for r in results)} total relations.")

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")

if __name__=="__main__" : 
    
  re_extractor = RelationExtractor(quantized_8=True)
  relations = re_extractor.extract_relations({
        "text": (
        "Satya Nadella is the CEO of Microsoft Corporation. He was born in Hyderabad and graduated from the University of Wisconsin. Microsoft was founded by Bill Gates."
      ),
        "entities": {
        "PERSON": ["Satya Nadella", "Bill Gates"],
        "ORG": ["Microsoft Corporation", "Microsoft", "University of Wisconsin"],
        "LOC": ["Hyderabad"]
      }
  })
  print(relations)