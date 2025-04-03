import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config as config 

class CollegePDFLoader:
    def __init__(self, pdf_dir=config.PDF_DIRECTORY):
        self.pdf_dir = pdf_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_pdfs(self):
        """Load all PDFs from the specified directory."""
        # Create PDF directory if it doesn't exist
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)
            print(f"Created PDF directory at {self.pdf_dir}")
            return []
            
        pdf_files = glob(os.path.join(self.pdf_dir, "*.pdf"))
        documents = []
        
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {pdf_file}: {e}")
        
        return self.text_splitter.split_documents(documents)