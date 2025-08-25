import requests,json, time ,re 
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from collections import Counter


''' Documentation 

* Class WikiDataAPI: 
    Functions :
    - search_wikidata_entities: Searches for entities in Wikidata using labels and aliases.
    - get_entity_aliases: Fetches aliases of an entity by its QID from Wikidata.

* Class EntityLinker:
    Functions :
    - extract_context_keywords: Extracts relevant keywords from context for better search.
    - candidate_search: Generates candidate entities using multiple search strategies.
    - string_similarities: Calculates string similarity scores between mention and candidates.
    - context_compatibility: Evaluates context compatibility between mention and candidates.
    - create_context_window: Creates a focused context window around the mention.
    - encode_mention: Encodes the mention with context windowing.
    - encode_candidates: Encodes the candidate entities for similarity comparison.
    - link_entities: Links entities using multi-stage ranking based on semantic, string, and context scores.

* Logic :

'''
class WikiDataAPI:
    @staticmethod
    def search_wikidata_entities(query, language='en', limit=10):
        """
        Perform a textual search on Wikidata using labels and aliases.
        """
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "search": query,
            "limit": limit
        }
        
        headers = {
            'User-Agent': 'AxeFinance-EntityLinker/1.0 (https://github.com/GhadaJeddey/AxeFinance) Python/requests'
        }
        
        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching data for query '{query}': {e}")
            return []
        
        results = []
        
        if response.status_code == 200:
            data = response.json()
            for entity in data.get("search", []):
                qid = entity.get("id")
                label = entity.get("label", "")
                description = entity.get("description", "")
                aliases = WikiDataAPI.get_entity_aliases(qid, language)
                
                results.append({
                    "qid": qid,
                    "label": label,
                    "description": description,
                    "aliases": aliases
                })
        
        return results
    
    @staticmethod
    def get_entity_aliases(qid, language='en'):
        """
        Fetch aliases of an entity by its QID from Wikidata.
        """
        if not qid:
            return []
            
        entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        
        headers = {
            'User-Agent': 'AxeFinance-EntityLinker/1.0 (https://github.com/GhadaJeddey/AxeFinance) Python/requests'
        }
        
        try:
            response = requests.get(entity_url, headers=headers, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            aliases = data['entities'][qid]['aliases'].get(language, [])
            return [alias['value'] for alias in aliases]
        except (KeyError, requests.RequestException):
            return []


class EntityLinker:
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        
        """
        Initialize the entity linker with improved features.
        """
        
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def extract_context_keywords(self, context, mention):
        
        """
        Extract the most relevant keywords from context for better search
        Args :
            context (str): The context text to extract keywords from.
            mention (str): The entity mention to focus on.
        Returns :
            List[str]: A list of the most relevant keywords in the context text.
        """
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'has', 
            'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might'
        }
        
        words = re.findall(r'\b[A-Za-z]{3,}\b', context.lower())
        mention_words = set(mention.lower().split())
        words = [w for w in words if w not in stop_words and w not in mention_words]

        word_freq = Counter(words) 

        return [word for word, _ in word_freq.most_common(3)]
    
    def candidate_search(self, mention_text, context, limit=15):
        
        """
        Candidate generation with multiple search strategies
        Args :
            mention_text (str): The entity mention text.
            context (str): The context in which the mention appears.
            limit (int): The maximum number of candidates to return.
        Returns :
            List[Dict[str, Any]]: A list of candidate entities.
        """
        
        all_candidates = []
        
        # 1. Direct mention search
        candidates1 = WikiDataAPI.search_wikidata_entities(mention_text, limit=limit//2)
        all_candidates.extend(candidates1)
        
        # 2. Search with context keywords 
        context_keywords = self.extract_context_keywords(context, mention_text)
        if context_keywords:
            query_with_context = f"{mention_text} {context_keywords[0]}"
            candidates2 = WikiDataAPI.search_wikidata_entities(query_with_context, limit=limit//3)
            all_candidates.extend(candidates2)
        
        # 3. Search with different casing
        if mention_text != mention_text.lower():
            candidates3 = WikiDataAPI.search_wikidata_entities(mention_text.lower(), limit=3)
            all_candidates.extend(candidates3)

        # Merge all candidates from all approaches
        seen_qids = set()
        unique_candidates = []
        for candidate in all_candidates:
            if candidate["qid"] not in seen_qids:
                unique_candidates.append(candidate)
                seen_qids.add(candidate["qid"])
        
        return unique_candidates[:limit]
    
    def create_context_window(self, mention_text, context, window_size=50):
        
        """
        Create a focused context window around the mention
        Args :
            mention_text (str): The entity mention text.
            context (str): The context in which the mention appears.
            window_size (int): The size of the context window to create.
        Returns :
            str: The context window around the mention.
        """

        mention_pattern = re.escape(mention_text.lower())  
        match = re.search(mention_pattern, context.lower())
        
        if not match:
            words = context.split()
            return " ".join(words[:window_size])
        
        words = context.split()
        mention_start_word = len(context[:match.start()].split())
        
        start_idx = max(0, mention_start_word - window_size//2)
        end_idx = min(len(words), mention_start_word + window_size//2)
        
        return " ".join(words[start_idx:end_idx])
    
    def encode_mention(self, mention_text, context):
        
        """ 
        Mention encoding with context windowing
        Args :
            mention_text (str): The entity mention text.
            context (str): The context in which the mention appears.
        Returns :
            torch.Tensor: The encoded representation of the mention with context.
        """
        
        context_window = self.create_context_window(mention_text, context)
        mention_with_context = f"{mention_text}. {context_window}"
        return self.model.encode(mention_with_context, convert_to_tensor=True)
    
    def encode_candidates(self, candidates):
        
        """
        Candidate encoding with more information
        Args :
            candidates (List[Dict[str, Any]]): The list of candidate entities.
        Returns :
            torch.Tensor: The encoded representations of the candidate entities :Each row in the tensor corresponds to one candidate entity.
        """
        
        texts = []
        
        for c in candidates:
            text_parts = []
            
            if c.get("label"):
                text_parts.append(c["label"])
            
            if c.get("aliases"):
                text_parts.extend(c["aliases"][:2])
            
            if c.get("description"):
                text_parts.append(c["description"])
            
            text = ". ".join(text_parts) if text_parts else "Unknown entity"
            texts.append(text)
        
        return self.model.encode(texts, convert_to_tensor=True) 
    
    def string_similarities(self, mention_text, candidates):
        
        """
        Calculate string-based similarity scores (similarity based on their character sequences)
        Args :
            mention_text (str): The entity mention text.
            candidates (List[Dict[str, Any]]): The list of candidate entities.
        Returns :
            List[float]: The list of similarity scores for each candidate.
        """
        
        scores = []
        mention_lower = mention_text.lower()
        
        for candidate in candidates:
            max_score = 0

            if candidate.get("label"):
                score = SequenceMatcher(None, mention_lower, candidate["label"].lower()).ratio()
                max_score = max(max_score, score)
            
            if candidate.get("aliases"):
                for alias in candidate["aliases"]:
                    score = SequenceMatcher(None, mention_lower, alias.lower()).ratio()
                    max_score = max(max_score, score)
            
            scores.append(max_score)
        
        return scores

    def context_compatibility(self, context, candidates):
        
        """
        Calculate context compatibility scores
        Args :
            context (str): The context in which the mention appears.
            candidates (List[Dict[str, Any]]): The list of candidate entities.
        Returns :
            List[float]: The list of context compatibility scores for each candidate.
        """
        
        scores = []
        context_keywords = set(self.extract_context_keywords(context, ""))
        
        for candidate in candidates:
            score = 0
            
            if candidate.get("description") and context_keywords:
                desc_words = set(candidate["description"].lower().split())
                overlap = len(context_keywords.intersection(desc_words))
                score = overlap / len(context_keywords) if context_keywords else 0
            
            scores.append(score)
        
        return scores

    def link_entities(self, candidates, mention_text, mention_context, top_k=1):
        
        """
        Entity linking with multi-stage ranking
        Args :
            candidates (List[Dict[str, Any]]): The list of candidate entities.
            mention_text (str): The entity mention text.
            mention_context (str): The context in which the mention appears.
            top_k (int): The number of top candidates to return.
        Returns :
            List[Dict[str, Any]]: The ranked list of linked entities.
            Dict[str, Any]: The best linked entity.
        """
        
        if not candidates:
            return []
        
        # Stage 1: Semantic similarity
        mention_embedding = self.encode_mention(mention_text, mention_context)
        candidate_embeddings = self.encode_candidates(candidates)
        
        semantic_scores = util.cos_sim(mention_embedding, candidate_embeddings)[0]
        
        # Stage 2: String similarity
        string_scores = self.string_similarities(mention_text, candidates)
        
        # Stage 3: Context compatibility
        context_scores = self.context_compatibility(mention_context, candidates)
        
        # Combine scores with weights
        combined_scores = []
        for i in range(len(candidates)):
            combined_score = (0.6 * semantic_scores[i].item() + 
                            0.3 * string_scores[i] + 
                            0.1 * context_scores[i])
            combined_scores.append(combined_score)
        
        top_results = []
        for score, idx in sorted(zip(combined_scores, range(len(candidates))), reverse=True)[:top_k]:
            candidate = candidates[idx]
            top_results.append({
                "qid": candidate["qid"],
                "label": candidate["label"],
                "description": candidate["description"],
                "score": score,
                "semantic_score": semantic_scores[idx].item(),
                "string_score": string_scores[idx],
                "context_score": context_scores[idx]
            })

        best_result = top_results[0] if top_results else None 
        
        return top_results , best_result
    
    def evaluation(self, data, top_k=3, delay=3, output_file="entity_linking_results.json"):
        
        """
        Evaluate the entity linker on the dataset.
        Args :
            data (List[Dict[str, Any]]): The dataset to evaluate.
            top_k (int): The number of top candidates to consider for evaluation.
            delay (int): The delay between processing samples (in seconds).
            output_file (str): The file to save the evaluation results.
        Returns :
            None
        """
        
        
        valid_samples = [sample for sample in data if sample["ground_truth_qid"] != "Q123456"]
        
        correct_predictions = 0
        total_score = 0.0
        all_results = []
        start_time = time.time()
        
        for i, sample in enumerate(data, 1):
            print(f"\n[{i}/{len(data)}] Processing: {sample['mention']}")
            
            if i > 1:
                time.sleep(delay)
            
            mention_text = sample["mention"]
            mention_context = sample["context"]
            true_qid = sample["ground_truth_qid"]
            
            
            candidates = self.candidate_search(mention_text, mention_context)
            
            linked_entities , best_result = self.link_entities(
                candidates, mention_text, mention_context, top_k=top_k
            )
            
            if linked_entities:
                best_entity = best_result
                best_qid = best_entity["qid"]
                best_score = best_entity["score"]
                is_correct = best_qid == true_qid
                
                if is_correct and true_qid != "Q123456":  
                    correct_predictions += 1
                    total_score += best_score
                    
            else:
                best_qid = None
                best_score = None
                is_correct = False

            
            all_results.append({
                "sample_id": i,
                "mention": mention_text,
                "context": mention_context,
                "ground_truth_qid": true_qid,
                "predicted_qid": best_qid,
                "is_correct": is_correct,
                "score": best_score,
                "top_k_predictions": linked_entities,
                "total_candidates": len(candidates)
            })
        
        end_time = time.time()
        
        valid_results = [r for r in all_results if r["ground_truth_qid"] != "Q123456"]
        accuracy = correct_predictions / len(valid_results) if valid_results else 0
        average_confidence = total_score / correct_predictions if correct_predictions > 0 else 0
        
        summary = {
            "model_name": self.model_name,
            "accuracy": accuracy,
            "average_confidence": average_confidence,
            "total_samples": len(data),
            "valid_samples": len(valid_results),
            "correct_predictions": correct_predictions,
            "total_time_seconds": end_time - start_time,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        output = {
            "summary": summary,
            "results": all_results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Model: {self.model_name}")
        print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{len(valid_results)})")
        print(f"Average Confidence: {average_confidence:.3f}")
        print(f"Total Time: {end_time - start_time:.1f} seconds")
        print(f"Results saved to: {output_file}")
        
        return output



def analyze_results(data,results_file="entity_linking_results.json"):
    """
    Analyze the evaluation results
    Args:
        data (Dict[str, Any]): The evaluation results data.
        results_file (str): The file to save the analysis results.
    """

    results = data["results"]
    summary = data["summary"]

    
    print(f"Overall Accuracy: {summary['accuracy']:.3f}")
    print(f"Average Confidence: {summary['average_confidence']:.3f}")
    
    errors = [r for r in results if not r["is_correct"] and r["ground_truth_qid"] != "Q123456"]
    
    print(f"\nError Analysis ({len(errors)} errors):")
    if errors:
        no_candidates = [r for r in errors if r["total_candidates"] == 0]
        low_confidence = [r for r in errors if r["score"] and r["score"] < 0.5]
        
        print(f"  - No candidates found: {len(no_candidates)}")
        print(f"  - Low confidence predictions: {len(low_confidence)}")
        
        print(f"\nExample errors:")
        for error in errors[:3]:
            print(f"  Mention: {error['mention']}")
            print(f"  Context: {error['context'][:60]}...")
            if error['top_k_predictions']:
                pred = error['top_k_predictions'][0]
                print(f"  Predicted: {pred['label']} ({pred['qid']}) - Score: {pred['score']:.3f}")
            else:
                print(f"  Predicted: No candidates")
            print(f"  Expected: {error['ground_truth_qid']}")
            print()

def main():
    """Main function to run the evaluation"""
    
    print("Entity Linker Evaluation")
    print("=" * 50)
    file_path = "evaluation_dataset.json"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {file_path}")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return []

    linker = EntityLinker("sentence-transformers/all-mpnet-base-v2")
    
    results = linker.evaluation(
        data=data,
        top_k=3,
        delay=5,
        output_file="entity_linking_results-20-08-2025.json"
    )

    analyze_results(data, "entity_linking_results.json")

import json
from typing import Dict, List, Tuple

def calculate_entity_linking_metrics(results_file_path: str) -> Dict[str, float]:
    """
    Calculate precision, recall, F1-score, and top-3 accuracy from entity linking results.
    
    Args:
        results_file_path (str): Path to the JSON results file
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """

    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    total_samples = len(results)
    
    correct_predictions = 0
    top_3_correct = 0
    
    for result in results:
        ground_truth_qid = result['ground_truth_qid']
        predicted_qid = result['predicted_qid']
        top_k_predictions = result['top_k_predictions']
        
        if predicted_qid == ground_truth_qid:
            correct_predictions += 1
        
        top_3_qids = [pred['qid'] for pred in top_k_predictions[:3]]
        if ground_truth_qid in top_3_qids:
            top_3_correct += 1
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    precision = accuracy  
    top_3_accuracy = top_3_correct / total_samples if total_samples > 0 else 0.0
    
    return {
        'precision': round(precision, 4),
        'top_3_accuracy': round(top_3_accuracy, 4),
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'top_3_correct': top_3_correct
    }

metrics = calculate_entity_linking_metrics("entity_linking_results.json")
print(metrics)