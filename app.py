
import streamlit as st
import requests
import json
import time
from typing import Optional
import sys

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
            response = requests.get(f"{self.api_base_url}/Summarizer/info", timeout=5)
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
                f"{self.api_base_url}/Summarizer/summarize",
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
            response = requests.get(f"{self.api_base_url}/NER/info", timeout=5)
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
            response = requests.post(f"{self.api_base_url}/NER/extract", json=data, timeout=60)
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
    def __init__(self, api_base_url="http://localhost:8001"):
        self.api_base_url = api_base_url

    def link_entity(self, mention, context, top_k=5, candidate_limit=15):
        try:
            data = {
                "mention": mention,
                "context": context,
                "top_k": top_k,
                "candidate_limit": candidate_limit
            }
            response = requests.post(f"{self.api_base_url}/EntityLinker/extract", json=data, timeout=60)
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
            response = requests.post(f"{self.api_base_url}/RE/extract", json=data, timeout=120)
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
    # Prompt for NER and RE API URLs at startup
    if not hasattr(st.session_state, "ner_api_url") or not hasattr(st.session_state, "re_api_url"):
        import getpass
        print("Please enter the NER API base URL (e.g., http://localhost:8001): ", end="", flush=True)
        ner_api_url = input().strip()
        print("Please enter the Relation Extraction API base URL (e.g., http://localhost:8003): ", end="", flush=True)
        re_api_url = input().strip()
        st.session_state["ner_api_url"] = ner_api_url
        st.session_state["re_api_url"] = re_api_url

    summarizer_app = SummarizationApp(api_base_url="http://localhost:8000")
    ner_app = NERApp(api_base_url=st.session_state["ner_api_url"])
    entity_app = EntityApp(api_base_url="http://localhost:8001")
    re_app = REApp(api_base_url=st.session_state["re_api_url"])

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


    # Use Streamlit containers for live updates
    summarizer_container = st.container()
    ner_container = st.container()
    entity_linking_container = st.container()
    re_container = st.container()

    summarization_result = None
    ner_result = None
    entity_linking_results = []
    re_result = None

    if run_jobs and input_text:
        # --- Summarization Section ---
        with summarizer_container:
            st.markdown('<h2 class="main-header">1. Summarizer</h2>', unsafe_allow_html=True)
            st.divider()
            with st.spinner("Summarizing text..."):
                summarization_success, summarization_result = summarizer_app.summarize_text(text=input_text)
            if summarization_success and isinstance(summarization_result, dict) and "summary" in summarization_result:
                st.markdown("#### 📄 Summary")
                st.text(summarization_result["summary"])
            elif summarization_success:
                st.markdown("#### 📄 Summary (Raw)")
                st.json(summarization_result)
            else:
                st.error(f"Summarization failed: {summarization_result}")

        # --- NER Section ---
        with ner_container:
            st.markdown('<h2 class="main-header">2. Named Entity Recognition</h2>', unsafe_allow_html=True)
            st.divider()
            with st.spinner("Extracting entities..."):
                ner_success, ner_result = ner_app.extract_entities(input_text)
            if ner_success and isinstance(ner_result, dict):
                st.markdown("#### 📋 Extracted Entities")
                st.json(ner_result)
            elif ner_success:
                st.markdown("#### 📋 Extracted Entities (Raw)")
                st.json(ner_result)
            else:
                st.error(f"NER failed: {ner_result}")

        # --- Entity Linking Section ---
        with entity_linking_container:
            st.markdown('<h2 class="main-header">3. Entity Linking</h2>', unsafe_allow_html=True)
            st.divider()
            entity_linking_results = []
            if ner_success and isinstance(ner_result, dict):
                relevant_types = [
                    "PERSON", "ORG", "WORK_OF_ART"
                ]
                ner_entities = ner_result["entities"] if "entities" in ner_result else ner_result
                for ent_type in relevant_types:
                    mentions = ner_entities.get(ent_type, [])
                    for mention in mentions:
                        with st.spinner(f"Linking entity: {mention} ({ent_type})"):
                            entity_success, entity_data = entity_app.link_entity(
                                mention=mention,
                                context=input_text,
                                top_k=3,
                                candidate_limit=10
                            )
                        result = {
                            "mention": mention,
                            "type": ent_type,
                            "success": entity_success,
                            "data": entity_data
                        }
                        entity_linking_results.append(result)
                        st.json(result)
                        st.divider()
                if not entity_linking_results:
                    st.info("No relevant entities found for linking.")
            else:
                st.info("Run all jobs to link entities from your text.")
                if ner_result:
                    st.warning("⚠️ NER found entities but Entity Linking didn't run. Check if the Entity Linking API is running on the correct port.")

        # --- Relation Extraction Section ---
        with re_container:
            st.markdown('<h2 class="main-header">4. Relation Extraction</h2>', unsafe_allow_html=True)
            st.divider()
            re_result = None
            if ner_success and isinstance(ner_result, dict):
                relevant_types = [
                    "PERSON", "ORG", "NATIONALITIES_RELIGIOUS_GROUPS", "LOC"
                ]
                ner_entities = ner_result["entities"] if "entities" in ner_result else ner_result
                re_entities = {k: v for k, v in ner_entities.items() if k in relevant_types}
                with st.spinner("Extracting relations..."):
                    re_success, re_result = re_app.extract_relations(input_text, re_entities)
                if re_success and isinstance(re_result, dict) and "relations" in re_result:
                    st.markdown("#### Extracted Relations")
                    st.json(re_result)
                elif re_success and isinstance(re_result, list) and len(re_result) > 0:
                    if isinstance(re_result[0], dict) and "type" in re_result[0] and re_result[0]["type"] == "list_type":
                        st.error("❌ Relation extraction failed - API validation error")
                        st.error("The RE API expects entities in a specific format. Please check the API documentation.")
                        with st.expander("🔍 View Error Details"):
                            st.json(re_result)
                    else:
                        st.json(re_result)
                elif re_success:
                    st.warning("⚠️ No relations extracted or unexpected response format")
                    with st.expander("🔍 View Raw Response"):
                        st.json(re_result)
                else:
                    st.error(f"Relation extraction failed: {re_result}")
            else:
                st.info("Run all jobs to extract relations from your text.")

if __name__ == "__main__":
    main()