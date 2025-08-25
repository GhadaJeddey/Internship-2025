
import streamlit as st
import requests
import json
import time
from typing import Optional
from neo4j_graph import Neo4jGraphManager

st.set_page_config(
    page_title="Jobs Dashboard",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
    }
    .summary-box {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background-color: #f1f5f9;
        border-radius: 0.4rem;
        margin: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

class SummarizationApp:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
    
    def check_api_health(self):
        """Check if the API is running and healthy"""
        try:
            response = requests.get(f"{self.api_base_url}/Summarizer/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            return False, str(e)
    
    def get_model_info(self):
        """Get model information from the API"""
        try:
            response = requests.get(f"{self.api_base_url}/SummarizerModel/info", timeout=5)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return None

    def summarize_text(self, text, max_length=500, min_length=100, num_beams=2, length_penalty=2.0):
        """Call the summarization API"""
        try:
            data = {
                "text": text,
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "length_penalty": length_penalty
            }
            
            response = requests.post(
                f"{self.api_base_url}/summarize",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.content else "API Error"
                return False, f"Error {response.status_code}: {error_detail}"
                
        except requests.exceptions.Timeout:
            return False, "Request timed out. The text might be too long or the server is busy."
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"

class NERApp:
    def __init__(self, api_base_url="http://localhost:8001"):
        self.api_base_url = api_base_url

    def get_model_info(self):
        """Get NER model information from the API"""
        try:
            response = requests.get(f"{self.api_base_url}/NERModel/info", timeout=5)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return None
        
    def check_api_health(self):
        """Check if the API is running and healthy"""
        try:
            response = requests.get(f"{self.api_base_url}/NER/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            return False, str(e)
        
    def extract_entities(self, text):
        try:
            data = {"text": text}
            response = requests.post(f"{self.api_base_url}/extract", json=data, timeout=60)
            if response.status_code == 200:
                return True, response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.content else "API Error"
                return False, f"Error {response.status_code}: {error_detail}"
        except requests.exceptions.Timeout:
            return False, "Request timed out. The text might be too long or the server is busy."
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        
class EntityApp:
    def __init__(self, api_base_url="http://localhost:8002"):
        self.api_base_url = api_base_url

    def link_entity(self, mention, context, top_k=5, candidate_limit=15):
        try:
            data = {
                "mention": mention,
                "context": context,
                "top_k": top_k,
                "candidate_limit": candidate_limit
            }
            response = requests.post(f"{self.api_base_url}/link", json=data, timeout=60)
            if response.status_code == 200:
                return True, response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.content else "API Error"
                return False, f"Error {response.status_code}: {error_detail}"
        except requests.exceptions.Timeout:
            return False, "Request timed out. The text might be too long or the server is busy."
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"

    def get_model_info(self):
        try:
            response = requests.get(f"{self.api_base_url}/EntityLinker/info", timeout=5)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return None

    def check_api_health(self):
        """Check if the API is running and healthy"""
        try:
            response = requests.get(f"{self.api_base_url}/EntityLinker/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            return False, str(e)
class REApp:
    def __init__(self, api_base_url="http://localhost:8003"):
        self.api_base_url = api_base_url

    def extract_relations(self, text, entities):
        try:
            data = {"text": text, "entities": entities}
            response = requests.post(f"{self.api_base_url}/RE/extract", json=data, timeout=60)
            if response.status_code == 200:
                return True, response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.content else "API Error"
                return False, error_detail
        except requests.exceptions.Timeout:
            return False, "Request timed out. The text might be too long or the server is busy."
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"

    def get_model_info(self):
        try:
            response = requests.get(f"{self.api_base_url}/RE/info", timeout=5)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return None

    def check_api_health(self):
        """Check if the API is running and healthy"""
        try:
            response = requests.get(f"{self.api_base_url}/RE/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            return False, str(e)
        
def main():


    # --- Neo4j Aura Graph Section ---
    st.markdown('<h2 class="main-header">5. Neo4j Aura Graph</h2>', unsafe_allow_html=True)
    st.divider()

    with st.sidebar:
        st.subheader("Neo4j Aura Credentials")
        neo4j_uri = st.text_input("Neo4j Aura URI", value="bolt://<your-neo4j-uri>")
        neo4j_user = st.text_input("Neo4j Username", value="neo4j")
        neo4j_password = st.text_input("Neo4j Password", type="password")
        show_graph = st.button("Show Neo4j Graph")

    graph_manager = None
    if neo4j_uri and neo4j_user and neo4j_password and "<your-neo4j-uri>" not in neo4j_uri:
        try:
            graph_manager = Neo4jGraphManager(neo4j_uri, neo4j_user, neo4j_password)
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}")

    if show_graph and graph_manager:
        try:
            graph_data = graph_manager.fetch_graph(limit=25)
            st.success("Connected to Neo4j Aura!")
            st.markdown("#### Sample Graph Data (first 25 relationships)")
            st.write("Nodes:", graph_data["nodes"])
            st.write("Edges:", graph_data["edges"])
            # Optional: visualize with networkx
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                G = nx.DiGraph()
                for node in graph_data["nodes"]:
                    G.add_node(node.get("id", str(node)), **node)
                for src, dst, rel in graph_data["edges"]:
                    G.add_edge(src, dst, label=rel)
                fig, ax = plt.subplots(figsize=(8, 5))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='skyblue', ax=ax)
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Graph visualization not available: {e}")
        except Exception as e:
            st.error(f"Failed to fetch or visualize Neo4j graph: {e}")

    summarizer_app = SummarizationApp(api_base_url="http://localhost:8000")
    ner_app = NERApp(api_base_url="http://localhost:8001")
    entity_app = EntityApp(api_base_url="http://localhost:8002")
    re_app = REApp(api_base_url="http://localhost:8003")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        st.subheader("API Status")
        if st.button("Check API Status", type="secondary"):
            with st.spinner("Checking API..."):
                summarizer_is_healthy, summarizer_health_info = summarizer_app.check_api_health()
                ner_is_healthy, ner_health_info = ner_app.check_api_health()
                entity_linking_is_healthy, entity_linking_health_info = entity_app.check_api_health()
                re_is_healthy, re_health_info = re_app.check_api_health()

            st.markdown("### API Health Status")
            st.markdown("#### Summarization API")
            if summarizer_is_healthy:
                st.success(" API is running!")
            else:
                st.error(" API is unavailable.")

            st.markdown("#### NER API")
            if ner_is_healthy:
                st.success(" API is running!")
            else:
                st.error(" API is unavailable.")

            st.markdown("#### Entity Linking API")
            if entity_linking_is_healthy:
                st.success(" API is running!")
            else:
                st.error(" API is unavailable.")
            
            st.markdown("#### Relation Extraction API")
            if re_is_healthy:
                st.success(" API is running!")
            else:
                st.error(" API is unavailable.")
            
            

        summarization_model_info = summarizer_app.get_model_info()
        if summarization_model_info:
            st.subheader("🧠 Summarization Model Info")
            st.info(f"**Model:** {summarization_model_info.get('model_name', 'N/A')}")
        
        st.divider()

        ner_model_info = ner_app.get_model_info()
        if ner_model_info:
            st.subheader("🧠 NER Model Info")
            st.info(f"**Model:** {ner_model_info.get('model_name', 'N/A')}\n**Entity Types:** {ner_model_info.get('entity_types', 'N/A')}")
        
        st.divider()

        el_model_info = entity_app.get_model_info()
        if el_model_info:
            st.subheader("🧠 Entity Linking Model Info")
            st.info(f"**Model:** {el_model_info.get('model_name', 'N/A')}\n**Supported Entity Types:** {el_model_info.get('supported_entity_types', 'N/A')}")
            
        re_model_info = re_app.get_model_info()
        if re_model_info:
            st.subheader("🧠 RE Model Info")
            st.info(f"**Model:** {re_model_info.get('model_name', 'N/A')}\n**Supported Relations:** {re_model_info.get('supported_relations', 'N/A')}\n**Supported Entity Types:** {re_model_info.get('supported_entity_types', 'N/A')}")

    st.markdown('<h1 class="main-header">Jobs Dashboard</h1>', unsafe_allow_html=True)
    st.divider()

    st.markdown('#### 📝 Input Article/Text')
    if st.button("Load Sample Text", type="secondary"):
        sample_text = """In the rapidly evolving landscape of artificial intelligence and machine learning, Microsoft Corporation announced on January 15th, 2024, that it would be investing approximately $2.5 billion over the next three years to establish five new research centers across the United States. The ambitious project, dubbed 'Project Phoenix', aims to revolutionize quantum computing capabilities and will employ over twelve thousand researchers and engineers. The first facility, located in Seattle, Washington, is expected to be operational by March 2025, with construction costs estimated at $450 million. CEO Satya Nadella emphasized during the 10:30 AM press conference that this represents the company's largest single investment in research and development since its founding in 1975. The initiative will collaborate with leading universities including Stanford University, MIT, and Carnegie Mellon University, creating partnerships worth an estimated $180 million annually. Industry experts predict this move could increase Microsoft's market share in the AI sector by approximately 15% within the next five years, potentially adding billions to the company's revenue stream."""
        st.session_state["unified_input_text"] = sample_text
        st.rerun()
    
    input_text = st.text_area(
        "Enter your article or text:",
        height=400,
        placeholder="Paste your long article or text here...",
        key="unified_input_text"
    )
    
    if input_text:
        char_count = len(input_text)
        word_count = len(input_text.split())
        st.caption(f"Characters: {char_count:,} | Words: {word_count:,}")
        if char_count < 50:
            st.warning("Text should be at least 50 characters long")
    
    st.divider()

    run_jobs = st.button(
        " Run All Jobs",
        type="primary",
        disabled=not input_text or len(input_text) < 50,
        use_container_width=True
    )

    summarization_result = None
    ner_result = None
    entity_linking_results = []
    re_result = None

    if run_jobs and input_text:

        with st.spinner("Summarizing text..."):
            summarization_success, summarization_result = summarizer_app.summarize_text(text=input_text)
        
        with st.spinner("Extracting entities..."):
            ner_success, ner_result = ner_app.extract_entities(input_text)
        
        # Entity Linking and Relation Extraction (if NER succeeded)
        if ner_success and isinstance(ner_result, dict):
            relevant_types = [
                "PERSON", "ORG", "LOC", "NATIONALITIES_RELIGIOUS_GROUPS", "LAW", "WORK_OF_ART"
            ]
            
            for ent_type in relevant_types:
                mentions = ner_result.get(ent_type, [])
                for mention in mentions:
                    with st.spinner(f"Linking entity: {mention} ({ent_type})"):
                        entity_success, entity_data = entity_app.link_entity(
                            mention=mention,
                            context=input_text,
                            top_k=3,
                            candidate_limit=10
                        )
                        entity_linking_results.append({
                            "mention": mention,
                            "type": ent_type,
                            "success": entity_success,
                            "data": entity_data
                        })
            
            # Relation Extraction (send full NER output)
            with st.spinner("Extracting relations..."):
                re_success, re_result = re_app.extract_relations(input_text, ner_result)
                # --- Update Neo4j Graph live ---
                if re_success and re_result and graph_manager:
                    try:
                        # Add nodes and relationships for each extracted relation
                        relations = re_result.get("relations") if isinstance(re_result, dict) else None
                        if relations and isinstance(relations, list):
                            for rel in relations:
                                # Example: rel = {"head": {"text": ..., "type": ...}, "tail": {...}, "relation": ...}
                                head = rel.get("head")
                                tail = rel.get("tail")
                                rel_type = rel.get("relation")
                                if head and tail and rel_type:
                                    # Add/find head node
                                    head_id = graph_manager.find_node(head.get("type", "Entity"), "text", head.get("text"))
                                    if not head_id:
                                        head_id = graph_manager.add_node(head.get("type", "Entity"), {"text": head.get("text")})
                                    # Add/find tail node
                                    tail_id = graph_manager.find_node(tail.get("type", "Entity"), "text", tail.get("text"))
                                    if not tail_id:
                                        tail_id = graph_manager.add_node(tail.get("type", "Entity"), {"text": tail.get("text")})
                                    # Add relationship
                                    if head_id and tail_id:
                                        graph_manager.add_relationship(head_id, tail_id, rel_type)
                        st.success("Neo4j graph updated with extracted relations.")
                    except Exception as e:
                        st.warning(f"Could not update Neo4j graph: {e}")

    # Display results sections
    
    # --- Summarization Section ---
    st.markdown('<h2 class="main-header">1. Summarizer</h2>', unsafe_allow_html=True)
    st.divider()
    
    if summarization_result:
        st.markdown("#### 📋 Generated Summary")
        st.markdown(f'''<div class="summary-box"><p>{summarization_result["summary"]}</p></div>''', unsafe_allow_html=True)
        st.markdown("#### Summary Statistics")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Compression", f"{summarization_result['compression_ratio']:.1%}")
        with col_b:
            st.metric("Summary Length", f"{summarization_result['summary_length']:,} chars")
        with st.expander("🔍 View Full API Response"):
            st.json(summarization_result)
    else:
        st.info("Run all jobs to generate a summary from your text.")

    # --- NER Section ---
    st.markdown('<h2 class="main-header">2. Named Entity Recognition</h2>', unsafe_allow_html=True)
    st.divider()
    
    if ner_result:
        st.markdown("#### 📋 Extracted Entities")
        st.json(ner_result)
    else:
        st.info("Run all jobs to extract entities from your text.")

    # --- Entity Linking Section ---
    st.markdown('<h2 class="main-header">3. Entity Linking</h2>', unsafe_allow_html=True)
    st.divider()
    
    if entity_linking_results:
        st.markdown("#### 📋 Entity Linking Results")
        for result in entity_linking_results:
            mention = result["mention"]
            ent_type = result["type"]
            entity_success = result["success"]
            entity_data = result["data"]
            
            st.markdown(f"**Mention:** `{mention}`")
            st.markdown(f"**Type:** `{ent_type}`")
            
            if entity_success:
                if isinstance(entity_data, dict):
                    candidates = entity_data.get("candidates") or entity_data.get("results")
                    if candidates and isinstance(candidates, list) and len(candidates) > 0:
                        top = candidates[0]
                        desc = top.get("description") or top.get("summary")
                        title = top.get("title") or top.get("label")
                        if title:
                            st.markdown(f"- **Top Candidate:** `{title}`")
                        if desc:
                            st.markdown(f"> {desc}")
                    else:
                        st.json(entity_data)
                else:
                    st.write(entity_data)
            else:
                st.error(f"Entity linking failed for `{mention}`: {entity_data}")
            
            st.divider()
    else:
        st.info("Run all jobs to link entities from your text.")

    # --- Relation Extraction Section ---
    st.markdown('<h2 class="main-header">4. Relation Extraction</h2>', unsafe_allow_html=True)
    st.divider()
    
    if re_result is not None:
        st.markdown("#### Extracted Relations")
        st.success("Relations extracted successfully!")
        st.json(re_result)

if __name__ == "__main__":
    main()