import os

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

class MedicalRAGService:
    def __init__(self, persist_directory="./chroma_db"):
        self.enabled = HAS_CHROMA
        if not self.enabled:
            print("ChromaDB not installed. RAG service will be bypassed.")
            return
            
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name="medical_guidelines")
            self._seed_data_if_empty()
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {e}")
            self.enabled = False
            
    def _seed_data_if_empty(self):
        """Seed the database with trusted WHO/CDC guidelines if empty."""
        if not self.enabled: return
        
        try:
            count = self.collection.count()
            if count == 0:
                print("Seeding ChromaDB with medical guidelines...")
                
                documents = [
                    "CDC Guideline: Normal Hemoglobin levels are 13.2-16.6 grams/dL for men and 11.6-15 grams/dL for women. Levels below 6.5 g/dL are critical and require immediate intervention.",
                    "WHO Guideline: Dengue fever is indicated by low platelet counts (often <100,000), Positive IgM (indicates acute infection), and Positive IgG (indicates past or secondary infection). Platelets below 20,000 is a severe risk of hemorrhage.",
                    "MedlinePlus: A normal platelet count ranges from 150,000 to 450,000 platelets per microliter of blood.",
                    "CDC: Seasonal Allergic Rhinitis is often treated with antihistamines. Normal blood tests will generally show no acute infection markers, though eosinophils may be elevated."
                ]
                
                ids = [f"doc_{i}" for i in range(len(documents))]
                metadatas = [
                    {"source": "CDC", "topic": "Hemoglobin"},
                    {"source": "WHO", "topic": "Dengue"},
                    {"source": "MedlinePlus", "topic": "Platelets"},
                    {"source": "CDC", "topic": "Allergies"}
                ]
                
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
        except Exception as e:
            print(f"Error seeding data: {e}")

    def retrieve_guidelines(self, queries: list, n_results=2) -> str:
        """Search ChromaDB using extracted entities."""
        if not self.enabled or not queries:
            return "No RAG evidence retrieved (ChromaDB disabled or empty query)."
            
        try:
            results = self.collection.query(
                query_texts=queries,
                n_results=n_results
            )
            
            evidence = []
            if results and 'documents' in results and results['documents']:
                for docs_list in results['documents']:
                    for doc in docs_list:
                        if doc not in evidence:
                            evidence.append(doc)
                            
            return "\n".join(evidence)
        except Exception as e:
            print(f"Error retrieving guidelines: {e}")
            return "Error retrieving guidelines."

# Global singleton instance
rag_service = MedicalRAGService()
