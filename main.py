import yaml
import json
import re
import numpy as np
import requests
from pathlib import Path
from src.data_processor import DataProcessor
from src.ner_model import NERModel
import time


def load_config(config_path):
    """Charge la configuration à partir d'un fichier YAML."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Le fichier {config_path} n'a pas été trouvé.")
        raise
    except yaml.YAMLError as e:
        print(f"Erreur lors du chargement du fichier YAML : {e}")
        raise

# Fonction pour déterminer si une entité doit être liée à Wikidata
def should_link_to_wikidata(entity_type, entity_text):
    """Détermine si une entité doit être liée à Wikidata"""
    
    linkable_types = {"PER", "ORG", "LOC", "GPE", "EVENT", "WORK_OF_ART", "PRODUCT", "FAC", "MISC"}
        
    non_linkable_types = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
    
    if entity_type in non_linkable_types:
        return False
    
    if entity_type in linkable_types:
        if len(entity_text.strip()) < 2:
            return False
        
        temporal_patterns = [
            r'this (week|month|year|day)',
            r'last (week|month|year|day)',
            r'next (week|month|year|day)',
            r'today|tomorrow|yesterday'
        ]
        
        for pattern in temporal_patterns:
            if re.match(pattern, entity_text.lower()):
                return False
        
        return True
    
    return False

def search_wikidata(entity_name, entity_type=None):
    """Recherche une entité sur Wikidata avec filtre intelligent par type"""

    url = "https://query.wikidata.org/sparql"
    entity_name = entity_name.strip()
    entity_name_escaped = entity_name.replace('"', '\\"')

    instance_map = {
        "PER": "wd:Q5",  
        "ORG": "wd:Q43229", 
        "LOC": "wd:Q618123",  
        "GPE": "wd:Q82794",   
        "PRODUCT": "wd:Q2424752", 
        "WORK_OF_ART": "wd:Q838948",  
        "EVENT": "wd:Q1190554",  
        "FAC": "wd:Q13226383",  
        "MISC": None  
    }

    instance_filter = ""
    if entity_type in instance_map and instance_map[entity_type]:
        instance_filter = f"?item wdt:P31/wdt:P279* {instance_map[entity_type]} ."

    query = f"""
    SELECT DISTINCT ?item ?itemLabel ?itemDescription WHERE {{
      {{
        ?item rdfs:label "{entity_name_escaped}"@en .
      }} UNION {{
        ?item skos:altLabel "{entity_name_escaped}"@en .
      }} UNION {{
        ?item rdfs:label "{entity_name_escaped}"@fr .
      }}
      {instance_filter}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,fr". }}
    }}
    LIMIT 10
    """

    headers = {
        'User-Agent': 'EntityLinkingBot/1.0 (https://example.com/contact)',
        'Accept': 'application/sparql-results+json'
    }

    params = {
        'format': 'json',
        'query': query
    }

    try:
        time.sleep(0.1)
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        results = response.json()

        if 'results' in results and 'bindings' in results['results']:
            candidates = []
            for result in results['results']['bindings']:
                label = result['itemLabel']['value']
                wikidata_id = result['item']['value'].split('/')[-1]
                description = result.get('itemDescription', {}).get('value', '')
                candidates.append({
                    'label': label,
                    'wikidata_id': wikidata_id,
                    'description': description
                })
            return candidates
        else:
            return []

    except Exception as e:
        print(f"[Erreur Wikidata] {entity_name}: {e}")
        return []




def filter_entities(entities, exclude_types=None):
    exclude_types = exclude_types or []
    
    filtered = []
    for e in entities:
      
        if (not re.match(r'^[.,!?;]$', e["word"]) and 
            e["score"] > 0.5 and 
            e["entity"] not in exclude_types and
            len(e["word"].strip()) > 1):  
            e["score"] = float(e["score"])
            filtered.append(e)
    
    return filtered



def convert_types(obj):
    """ Convertit récursivement les types non-JSON-sérialisables (comme float32) """
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(elem) for elem in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def deduplicate_entities(entities):
    """Supprime les entités dupliquées en gardant celle avec le meilleur score."""
    entity_map = {}
    
    for entity in entities:
        key = (entity["word"].lower(), entity["entity"])
        if key not in entity_map or entity["score"] > entity_map[key]["score"]:
            entity_map[key] = entity
    
    return list(entity_map.values())


def main():
    print("=== Démarrage du système d'Entity Linking ===")
    
    try:
        config = load_config("config/config.yaml")
        ner_model = NERModel(config)
        processor = DataProcessor(config["data"]["raw_path"])
        raw_path = Path(config["data"]["raw_path"])
        raw_path.mkdir(parents=True, exist_ok=True)

        # Chargement des articles
        articles = processor.load_articles(config["data"]["articles_file"], config["data"].get("separator"))
        print(f"Chargement de {len(articles)} article(s)")
        
        # Initialisation des compteurs
        entity_counts = {k: 0 for k in [
            "PER", "ORG", "LOC", "MISC", "CARDINAL", "DATE", "EVENT", "FAC",
            "GPE", "LANGUAGE", "LAW", "MONEY", "NORP", "ORDINAL", "PERCENT",
            "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"
        ]}

        all_results = []
        total_entities_linked = 0

        # Traitement des articles
        for idx, article in enumerate(articles, 1):
            print(f"\n--- Traitement article {idx}/{len(articles)} ---")
            
            segments = processor.segment_text(article, config["model"]["max_length"])
            article_entities = []
            offset = 0

            # Extraction des entités par segment
            for seg in segments:
                try:
                    results = ner_model.predict(seg)
                    for ent in results:
                        ent['start'] += offset
                        ent['end'] += offset
                        
                    filtered = filter_entities(results, config["data"].get("exclude_types", []))
                    article_entities.extend(filtered)
                except Exception as e:
                    print(f"Erreur segment {idx} : {e}")
                offset += len(seg) + 1

            # Déduplication des entités
            article_entities = deduplicate_entities(article_entities)
            print(f"Entités trouvées : {len(article_entities)}")

            # Entity linking vers Wikidata
            entities_linked_in_article = 0
            for ent in article_entities:
                entity_text = ent["word"].strip()
                entity_type = ent["entity"]
                
                # Vérifier si l'entité doit être liée à Wikidata
                if should_link_to_wikidata(entity_type, entity_text):
                    print(f"  Recherche Wikidata pour: {entity_text} ({entity_type})")
                    wikidata_candidates = search_wikidata(entity_text, entity_type)
                    
                    if wikidata_candidates:
                        ent["wikidata_candidates"] = [c['label'] for c in wikidata_candidates]
                        ent["wikidata_id"] = wikidata_candidates[0]['wikidata_id']
                        ent["wikidata_description"] = wikidata_candidates[0]['description']
                        ent["mapped_id"] = wikidata_candidates[0]['wikidata_id']
                        print(f"    -> Trouvé: {wikidata_candidates[0]['wikidata_id']} - {wikidata_candidates[0]['label']}")
                        entities_linked_in_article += 1
                        total_entities_linked += 1
                    else:
                        ent["wikidata_candidates"] = []
                        ent["wikidata_id"] = None
                        ent["mapped_id"] = "N/A"
                        print(f"    -> Aucun candidat trouvé")
                else:
                    # Pour les entités qui ne doivent pas être liées (DATE, TIME, etc.)
                    ent["wikidata_candidates"] = []
                    ent["wikidata_id"] = None
                    ent["mapped_id"] = "N/A"
                    print(f"  Entité non liée: {entity_text} ({entity_type})")

            print(f"  Entités liées à Wikidata: {entities_linked_in_article}")

            # Mise à jour des compteurs
            for ent in article_entities:
                entity_counts[ent["entity"]] += 1

            all_results.append({
                "article_id": idx,
                "text": article,
                "entities": article_entities,
                "entity_count": len(article_entities),
                "linked_entities": entities_linked_in_article
            })

        # Sauvegarde des résultats
        print(f"\n=== Sauvegarde des résultats ===")
        output_file = raw_path / config["data"]["output_file"]
        
        with open(output_file, "w", encoding="utf-8") as f:
            for result in all_results:
                f.write(f"\n=== Article {result['article_id']} ===\n")
                f.write(f"Texte : {result['text'][:100]}...\n")
                f.write(f"Nombre d'entités : {result['entity_count']}\n")
                f.write(f"Entités liées à Wikidata : {result['linked_entities']}\n")
                f.write("Entités détectées :\n")
                
                for ent in result["entities"]:
                    ctx = processor.get_entity_context(result["text"], ent["start"], ent["end"])
                    f.write(f"Entité : {ent['word']} | Type : {ent['entity']} | Score : {ent['score']:.3f} | Contexte : {ctx}\n")
                    
                    if ent.get("wikidata_id"):
                        f.write(f"  -> Wikidata ID : {ent['wikidata_id']}\n")
                        if ent.get("wikidata_description"):
                            f.write(f"  -> Description : {ent['wikidata_description']}\n")
                    else:
                        f.write(f"  -> Mapped ID : {ent['mapped_id']}\n")
                    
                    if ent.get("wikidata_candidates"):
                        candidates_str = ', '.join(ent['wikidata_candidates'][:3])  
                        f.write(f"  -> Wikidata Candidates : {candidates_str}\n")
                    else:
                        f.write(f"  -> Wikidata Candidates : Aucun\n")
                    f.write("\n")
                
                if not result["entities"]:
                    f.write("Aucune entité détectée.\n")

            f.write("\n=== Résumé global ===\n")
            f.write(f"Total articles : {len(all_results)}\n")
            f.write(f"Total entités : {sum(entity_counts.values())}\n")
            f.write(f"Total entités liées à Wikidata : {total_entities_linked}\n")
            f.write(f"Pourcentage d'entités liées : {(total_entities_linked / sum(entity_counts.values()) * 100):.1f}%\n")
            f.write("\nRépartition par type :\n")
            for k, v in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                if v > 0:
                    f.write(f"{k} : {v}\n")

        # Sauvegarde JSON
        if config["data"].get("save_json", False):
            json_file = raw_path / "ner_results.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(convert_types(all_results), f, ensure_ascii=False, indent=2)
            print(f"Résultats JSON sauvegardés : {json_file}")

            # Sauvegarde des statistiques
            stats_file = raw_path / "statistics.json"
            stats = {
                "total_articles": len(all_results),
                "total_entities": sum(entity_counts.values()),
                "total_entities_linked": total_entities_linked,
                "linking_percentage": (total_entities_linked / sum(entity_counts.values()) * 100) if sum(entity_counts.values()) > 0 else 0,
                "entity_counts": entity_counts,
                "average_entities_per_article": sum(entity_counts.values()) / len(all_results) if all_results else 0
            }
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"Statistiques sauvegardées : {stats_file}")

        print(f"\n=== Traitement terminé ===")
        print(f"{len(all_results)} article(s) analysé(s)")
        print(f"Total entités détectées : {sum(entity_counts.values())}")
        print(f"Entités liées à Wikidata : {total_entities_linked}")
        print(f"Résultats sauvegardés : {output_file}")

    except Exception as e:
        print(f"Erreur dans le traitement principal: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
  
    
