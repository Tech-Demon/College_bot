from sqlalchemy import create_engine, MetaData, inspect
from langchain.docstore.document import Document
import config as config

class CollegeDatabaseLoader:
    def __init__(self, db_url=config.DATABASE_URL):
        self.engine = create_engine(db_url)
        self.metadata = MetaData()
        self.inspector = inspect(self.engine)
        
    def get_tables(self):
        """Get all table names from the database."""
        return self.inspector.get_table_names()
    
    def get_schema_info(self):
        """Get schema information for all tables."""
        tables = self.get_tables()
        schema_docs = []
        
        for table in tables:
            columns = self.inspector.get_columns(table)
            column_info = [f"{col['name']} ({col['type']})" for col in columns]
            
            content = f"Table: {table}\nColumns: {', '.join(column_info)}"
            schema_docs.append(Document(page_content=content, metadata={"source": f"db_schema_{table}"}))
        
        return schema_docs
    
    def query_database(self, query):
        """Execute SQL query and return results."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return result.fetchall()
        except Exception as e:
            return f"Error executing query: {e}"