from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
import config as config

class CollegeBotAgent:
    def __init__(self, web_retriever, pdf_retriever, db_interface):
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0
        )
        self.web_retriever = web_retriever
        self.pdf_retriever = pdf_retriever
        self.db_interface = db_interface
        
        # Set up tools
        self.tools = self._create_tools()
        
        # Set up agent
        self.agent = self._create_agent()
        
    def _create_tools(self):
        """Create tools for the agent to use."""
        web_tool = create_retriever_tool(
            self.web_retriever,
            "website_search",
            "Search for information on the college website."
        )
        
        pdf_tool = create_retriever_tool(
            self.pdf_retriever,
            "document_search",
            "Search through college PDF documents like brochures, handbooks, etc."
        )
        
        def query_database(query):
            """Execute SQL query on the college database."""
            return self.db_interface.query_database(query)
        
        db_tool = Tool(
            name="database_query",
            func=query_database,
            description="Run SQL queries against the college database. Use this for structured data like courses, faculty, events, etc."
        )
        
        return [web_tool, pdf_tool, db_tool]
    
    def _create_agent(self):
        """Create a LangChain agent with the tools."""
        system_message = """You are an AI assistant for a college website. 
        You have access to the following tools:
        - website_search: Search through the college website content
        - document_search: Search through college PDF documents
        - database_query: Run SQL queries on the college database
        
        Always try to provide accurate, helpful information based on the college's data.
        If you don't know something, say so rather than making up information.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        react_agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5
        )
    
    async def process_query(self, query, chat_history=None):
        """Process a user query and return the response."""
        if chat_history is None:
            chat_history = []
            
        response = await self.agent.ainvoke({
            "input": query,
            "chat_history": chat_history
        })
        
        return response["output"]