from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient
import config as config

class MongoDBVectorStore:
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DB_NAME]
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY
        )
        
    def create_from_documents(self, documents, collection_name):
        """Create a vector store from documents."""
        collection = self.db[collection_name]
        
        # Clear existing data
        collection.delete_many({})
        
       # Check for vector search index existence
        index_name = f"{collection_name}_vector_index"
        try:
            # Check if index exists
            existing_indexes = list(collection.list_indexes())
            if not any(index["name"] == index_name for index in existing_indexes):
                print(f"Warning: Vector search index '{index_name}' does not exist. Please create it in MongoDB Atlas with the following JSON:")
                print("""
                {
                "name": "embedding_index",
                "type": "vectorSearch",
                "fields": [
                    {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 768,
                    "similarity": "cosine"
                    }
                ]
                }
                """)
            else:
                print(f"Vector search index '{index_name}' already exists.")
        except Exception as e:
            print(f"Warning: Error checking indexes - {e}")

        # Create vector store
        return MongoDBAtlasVectorSearch.from_documents(
            documents,
            self.embeddings,
            collection=collection,
            index_name=index_name
        )
    
    def load_vector_store(self, collection_name):
        """Load an existing vector store."""
        collection = self.db[collection_name]
        index_name = f"{collection_name}_vector_index"
        
        return MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.embeddings,
            index_name=index_name,
        )