from sqlalchemy import create_engine, MetaData, inspect, text
from langchain.docstore.document import Document
import config as config
import pymysql

class CollegeDatabaseLoader:
    def __init__(self, db_url=config.DATABASE_URL):
        # Create engine with appropriate settings for MySQL
        self.engine = create_engine(
            db_url, 
            pool_recycle=3600,  # Prevent connection timeouts
            pool_pre_ping=True  # Verify connections before using
        )
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
                # Use parameterized query to prevent SQL injection
                result = conn.execute(text(query))
                rows = result.fetchall()
                if not rows:
                    return "Query executed successfully. No results returned."
                    
                # Convert to list of dictionaries for easier processing
                column_names = result.keys()
                results = [dict(zip(column_names, row)) for row in rows]
                return results
                
        except Exception as e:
            return f"Error executing query: {e}"
    
    def test_connection(self):
        """Test if the database connection is working."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            return f"Connection error: {e}"