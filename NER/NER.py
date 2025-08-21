import re , string, warnings , torch , string, json_repair , json,os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,AutoModelForTokenClassification
from collections import defaultdict
from pydantic import BaseModel, Field, validator , root_validator, model_validator
from typing import Dict, List, Tuple, Set, Optional, Any , Dict
from pprint import pprint
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re , string, warnings , torch , string, json_repair , json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,AutoModelForTokenClassification
from collections import defaultdict
from pydantic import BaseModel, Field, validator , root_validator, model_validator
from typing import Dict, List, Tuple, Set, Optional, Any , Dict
from pprint import pprint
import re , string, warnings , torch , string, json_repair , json,os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,AutoModelForTokenClassification
from collections import defaultdict
from pydantic import BaseModel, Field, validator , root_validator, model_validator
from typing import Dict, List, Tuple, Set, Optional, Any , Dict
from pprint import pprint
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class NEREntities(BaseModel):
    PERSON: List[str] = Field(default_factory=list, description="People names (e.g., 'Elon Musk', 'Dr. Smith')")
    ORG: List[str] = Field(default_factory=list, description="Companies/organizations ('Apple', 'NASA', 'University of Paris')")
    LOC: List[str] = Field(default_factory=list, description="Countries, cities or geographic locations, features ('France', 'Paris', 'California' , 'Amazon rainforest')")
    WORK_OF_ART: List[str] = Field(default_factory=list, description="Titled creative works with proper names (books, films, songs, paintings, etc.).")
    NATIONALITIES_RELIGIOUS_GROUPS : List[str] = Field(default_factory=list, description="Nationalities or religious groups (e.g., 'American', 'Spanish', 'Catholics')")
    DATE: List[str] = Field(default_factory=list, description="Dates or time periods (e.g., '2023', 'next week')")
    TIME: List[str] = Field(default_factory=list, description="Specific times (e.g., 'midnight', '3:30 PM')")
    MONEY: List[str] = Field(default_factory=list, description="Monetary values (e.g., '$5', '€50 million' , '50 cents','ten millin dollars')")
    PERCENT: List[str] = Field(default_factory=list, description="Percentages (e.g., '%','3 percent', 'ninety-nine percent','0.1 percentage')")
    LAW: List[str] = Field(default_factory=list, description="Legal documents (e.g., 'Constitution', 'Civil Rights Act')")

    @validator('*', pre=True)
    def ensure_list(cls, v):
        """Ensures all fields are lists"""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        return []

    @classmethod
    def get_field_info(cls) -> Dict[str, Dict[str, Any]]:
        """Extracts field information for prompt generation"""
        field_info = {}
        for field_name, field in cls.__fields__.items():
            field_info[field_name] = {
                'name': field_name,
                'description': field.description or '',
                'type': str(field.annotation),
                'required': field.is_required()
            }
        return field_info


class NERExtractor:

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",quantized_8:bool=False) -> None:
        self.model_name = model_name
        self.tokenizer =  AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if quantized_8: 
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True
            )
        else :
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )

    def generate_system_prompt_from_schema(self) -> str:

            field_info = NEREntities.get_field_info()

            entity_descriptions = []
            for field_name, info in field_info.items():
                required_status = "REQUIRED" if info['required'] else "OPTIONAL"
                entity_descriptions.append(
                    f"- {info['name']}: {info['description']} "
                    f"[Type: {info['type']}, Status: {required_status}]"
                )

            system_prompt = f"""
You are a comprehensive Named Entity Recognition system. Extract ALL types of named entities with high precision. Focus on entities that start with a **capital letter** .
Return a JSON object that matches the provided Pydantic schema.
ENTITY TYPES :
{chr(10).join(entity_descriptions)}

Extract entities as they are in the original text , do not change them.
If a category is not present, **return an empty list**.

    """
            return system_prompt


    def extract_entities(self, texte: str) -> NEREntities:
        """
        Extract named entities using Pydantic schema validation
        """
        user_prompt = f"""Extract named entities from the following text.

Text:
\"\"\"{texte}\"\"\"
"""
        system_prompt = self.generate_system_prompt_from_schema()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_tokens = self.tokenizer.tokenize(system_prompt +'\n'+user_prompt)
            print(f"tokens length : {len(input_tokens)}")
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=1000,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    temperature=0.1
                )

            generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            try:
                start_idx = result.find('{')
                end_idx = result.rfind('}')
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No JSON found in response")

                json_str = result[start_idx:end_idx+1]
                raw_entities = json.loads(json_str)
                return NEREntities(**raw_entities)

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Retrying with JSON repair... Error: {e}")
                repaired = json_repair.repair_json(result)
                return NEREntities(**json.loads(repaired))

        except Exception as e:
            print(f"Entity extraction failed: {str(e)}")
            return NEREntities()

    def post_process_entities(self, entities: NEREntities, original_text: str) -> NEREntities:

        corrected: Dict[str, List[str]] = defaultdict(list)
        text_lower = original_text.lower()

        strip_chars = string.punctuation.replace("$", "").replace("%", "") + """''"""

        for field, values in entities.dict().items():
          for val in values:

              val_clean = val.strip(strip_chars).strip()

              if re.fullmatch(r"[\w\s\-'']+", val_clean):
                  pattern = r'\b' + re.escape(val_clean) + r'\b'
              else:
                  pattern = re.escape(val_clean)

              if re.search(pattern, original_text, flags=re.IGNORECASE):
                  corrected[field].append(val_clean)



        hallucinated_locs = []
        if "LOC" in corrected and "NATIONALITIES_RELIGIOUS_POLITICAL_GROUPS" in corrected :
            for norp in corrected["NATIONALITIES_RELIGIOUS_POLITICAL_GROUPS"]:
                if norp in corrected["LOC"]:
                    hallucinated_locs.append(norp)
            corrected["LOC"] = [loc for loc in corrected["LOC"] if loc not in hallucinated_locs]

        for field in corrected:
            filtered = []
            seen = set()
            for val in corrected[field]:
                if (
                    val
                    and val.lower() in text_lower
                    and (
                        not val[0].isalpha()
                        or val[0].isupper()
                        or re.search(r'\d', val)
                    )
                    and val not in seen
                ):
                    filtered.append(val)
                    seen.add(val)
            corrected[field] = filtered

        return NEREntities(**corrected)


    def token_counter(self, text,tokenizer) :
        tokens = tokenizer.tokenize(text)
        return len(tokens)

    def split_into_sentences(self,text):

        sentences = []
        start = 0

        for match in re.finditer(r'\.', text):
            end_idx = match.end()

            before_point = text[max(0, match.start()-5):match.start()]
            before_point = before_point.strip()
            last_word = before_point.split()[-1] if before_point else ""

            if len(last_word) > 2:
                after_point = text[match.end():match.end()+2]
                if re.match(r'\s+[A-Z]', after_point):
                    sentences.append(text[start:end_idx].strip())
                    start = end_idx

        if start < len(text):
            sentences.append(text[start:].strip())

        return sentences

    def chunk_text(self, text, tokenizer, max_tokens=500, tolerance=0.1):

        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        max_allowed = max_tokens * (1 + tolerance)

        for sent in sentences:
            sent_tokens = self.token_counter(sent, tokenizer)

            if current_tokens + sent_tokens > max_allowed:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


    def merge_entities(self, entities1: NEREntities, entities2: NEREntities) -> NEREntities:
        merged = {}
        for field in entities1.model_fields:
            merged[field] = entities1.__getattribute__(field) + entities2.__getattribute__(field)
        return NEREntities(**merged)

    def ner_predict(self, text: str) -> NEREntities:
        print('HERE IN NER PREDICT')
        max_tokens = 300
        if self.token_counter(text, self.tokenizer) > max_tokens:

            chunks = self.chunk_text(text, self.tokenizer)
            entities = NEREntities()
            for i, chunk in enumerate(chunks) :
                entities_chunk = self.extract_entities(chunk)
                entities = self.merge_entities(entities,entities_chunk)

        else :
            entities = self.extract_entities(text)

        entities = self.post_process_entities(entities,text)
        res = entities.model_dump(exclude_defaults=True)
        print(json.dumps(res,indent=2,ensure_ascii=False))
        return json.dumps(res,indent=2,ensure_ascii=False)

    def evaluation(self, dataset_path: str = "evaluation.json",
                  results_file: str = "evaluation_results.json",
                  predictions_file: str = "predictions.json") -> Dict[str, Any]:
        """
        Evaluate the NER model on the evaluation dataset.
        Loads evaluation.json, runs predictions, and compares to ground truth.
        Returns classification report and confusion matrix, saves results to file.
        """

        with open(dataset_path, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)

        y_true = []
        y_pred = []
        all_predictions = []

        all_pred_labels = set()
        all_true_labels = set()

        for idx, sample in enumerate(evaluation_data):
            text = sample['text']
            true_entities = sample['entities']

            pred_json = self.ner_predict(text)
            pred_entities_dict = json.loads(pred_json)

            for label in pred_entities_dict.keys():
                if pred_entities_dict[label]:  
                    all_pred_labels.add(label)
                    
            true_entities_dict = defaultdict(list)
            for entity in true_entities:
                if 'text' in entity and 'label' in entity:
                    label = entity['label']
                    all_true_labels.add(label)

                    if label in ['PERSON', 'ORG', 'LOC', 'WORK_OF_ART', 'NATIONALITIES_RELIGIOUS_GROUPS',
                              'DATE', 'TIME', 'MONEY', 'PERCENT', 'LAW']:
                        true_entities_dict[label].append(entity['text'])

            true_entities_dict = dict(true_entities_dict)

            y_true.append(true_entities_dict)
            y_pred.append(pred_entities_dict)

            sample_prediction = {
                'text': text,
                'true_entities': true_entities,
                'predicted_entities': pred_entities_dict,
                'sample_id': idx
            }
            all_predictions.append(sample_prediction)

            if idx < 3:
                print(f"\nSample {idx}:")
                print(f"Text: {text[:100]}...")
                print(f"True entities: {true_entities_dict}")
                print(f"Pred entities: {pred_entities_dict}")

        print(f"\nAll true labels found: {all_true_labels}")
        print(f"All pred labels found: {all_pred_labels}")

        results = self._compute_ner_metrics(y_true, y_pred)

        results['evaluation_metadata'] = {
            'dataset_size': len(evaluation_data),
            'evaluation_date': datetime.now().isoformat(),
            'model_name': self.model_name,
            'all_true_labels': list(all_true_labels),
            'all_pred_labels': list(all_pred_labels)
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {results_file}")
        print(f"Predictions saved to: {predictions_file}")

        self._print_formatted_results(results)

        return results

    def _compute_ner_metrics(self, y_true: List[Dict[str, List[str]]], y_pred: List[Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Compute NER evaluation metrics including classification report and confusion matrix.
        """

        all_types = set()
        for sample in y_true + y_pred:
            all_types.update(sample.keys())
        all_types = sorted(all_types)

        print(f"Entity types being evaluated: {all_types}")

        tp = Counter()  
        fp = Counter()  
        fn = Counter()  
        support = Counter()  
        confusion = defaultdict(lambda: Counter())  

        
        pred_count = Counter()  

        for sample_idx, (true_sample, pred_sample) in enumerate(zip(y_true, y_pred)):

            true_entities = set()
            for t, ents in true_sample.items():
                for e in ents:
                    true_entities.add((e.lower().strip(), t)) 
                    support[t] += 1

            pred_entities = set()
            for t, ents in pred_sample.items():
                for e in ents:
                    pred_entities.add((e.lower().strip(), t))  
                    pred_count[t] += 1

 
            for ent in pred_entities & true_entities:
                tp[ent[1]] += 1
                confusion[ent[1]][ent[1]] += 1

            for ent in pred_entities - true_entities:
                fp[ent[1]] += 1
                matched = False
                for true_ent in true_entities:
                    if true_ent[0] == ent[0] and true_ent[1] != ent[1]:
                        confusion[true_ent[1]][ent[1]] += 1
                        matched = True
                        break
                if not matched:
                    confusion['O'][ent[1]] += 1 

            for ent in true_entities - pred_entities:
                fn[ent[1]] += 1
                matched = False
                for pred_ent in pred_entities:
                    if pred_ent[0] == ent[0] and pred_ent[1] != ent[1]:
                        confusion[ent[1]][pred_ent[1]] += 1
                        matched = True
                        break
                if not matched:
                    confusion[ent[1]]['O'] += 1  


        report = {}
        for t in all_types:
            p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
            r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            report[t] = {
                'precision': round(p, 4),
                'recall': round(r, 4),
                'f1-score': round(f1, 4),
                'support': support[t],
                'pred_count': pred_count[t],
                'correct': tp[t]
            }

            print(f"{t}: TP={tp[t]}, FP={fp[t]}, FN={fn[t]}, Support={support[t]}, Pred={pred_count[t]}")

       
        if all_types:
            macro_p = np.mean([report[t]['precision'] for t in all_types])
            macro_r = np.mean([report[t]['recall'] for t in all_types])
            macro_f1 = np.mean([report[t]['f1-score'] for t in all_types])

            total_support = sum(support.values())
            if total_support > 0:
                weighted_p = sum(report[t]['precision'] * support[t] for t in all_types) / total_support
                weighted_r = sum(report[t]['recall'] * support[t] for t in all_types) / total_support
                weighted_f1 = sum(report[t]['f1-score'] * support[t] for t in all_types) / total_support
            else:
                weighted_p = weighted_r = weighted_f1 = 0.0

            total_correct = sum(tp.values())
            total_predictions = sum(pred_count.values())
            accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

            report['macro avg'] = {
                'precision': round(macro_p, 4),
                'recall': round(macro_r, 4),
                'f1-score': round(macro_f1, 4),
                'support': int(total_support)
            }

            report['weighted avg'] = {
                'precision': round(weighted_p, 4),
                'recall': round(weighted_r, 4),
                'f1-score': round(weighted_f1, 4),
                'support': int(total_support)
            }

            report['accuracy'] = {
                'precision': round(accuracy, 4),
                'recall': round(accuracy, 4),
                'f1-score': round(accuracy, 4),
                'support': int(total_support)
            }

        confusion_matrix = {t: dict(confusion[t]) for t in confusion}

        return {'classification_report': report, 'confusion_matrix': confusion_matrix}

    def _print_formatted_results(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation results in the specified format.
        """
        report = results['classification_report']

        if 'macro avg' in report:
            global_p = report['macro avg']['precision']
            global_r = report['macro avg']['recall']
            global_f1 = report['macro avg']['f1-score']

            print("\n" + "="*80)
            print("**GLOBAL METRICS:**")
            print(f"**Precision: {global_p:.4f}, Recall: {global_r:.4f}, F1: {global_f1:.4f}**")
            print("="*80)

        print("**PER-ENTITY STATS:**")
        print("**Entity              True     Pred  Correct        P        R       F1**")
        print("**" + "-"*80 + "**")

        entity_types = [k for k in report.keys() if k not in ['macro avg', 'weighted avg', 'accuracy']]
        entity_types.sort()

        for entity in entity_types:
            stats = report[entity]
            true_count = stats['support']
            pred_count = stats.get('pred_count', 0)
            correct = stats.get('correct', 0)
            precision = stats['precision']
            recall = stats['recall']
            f1 = stats['f1-score']

            print(f"**{entity:<20} {true_count:>5} {pred_count:>8} {correct:>8} {precision:>7.4f} {recall:>8.4f} {f1:>8.4f}**")

        print("**" + "="*80 + "**")


    def plot_confusion_matrix(self, results: Dict[str, Any], figsize=(12, 10), cmap="Blues"):
        """
        Draws a confusion matrix heatmap from the evaluation results.

        Args:
            results (Dict[str, Any]): Output of `evaluation` containing 'confusion_matrix'.
            figsize (tuple): Figure size for the plot.
            cmap (str): Color map for the heatmap.
        """
        confusion_matrix = results.get("confusion_matrix", {})

        labels = sorted(confusion_matrix.keys())

        matrix = []
        for true_label in labels:
            row = []
            for pred_label in labels:
                row.append(confusion_matrix[true_label].get(pred_label, 0))
            matrix.append(row)

        plt.figure(figsize=figsize)
        sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"NER Confusion Matrix ({self.model_name})")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

