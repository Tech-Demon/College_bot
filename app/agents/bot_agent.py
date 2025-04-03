from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import config as config
import logging

logging.basicConfig(level=logging.INFO)

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
            result = self.db_interface.query_database(query)
            return str(result)  # Return as a message
        
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

        {tools}
        
        Always try to provide accurate, helpful information based on the college's data.
        If you don't know something, say so rather than making up information.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Add tool_names to the prompt variables
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))
        
        react_agent = create_react_agent(
            self.llm,
            self.tools,
            prompt
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=react_agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    async def process_query(self, query, chat_history=None):
        """Process a user query and return the response."""
        if chat_history is None:
            chat_history = []
            
        # Manually format agent_scratchpad
        def _format_scratchpad(intermediate_steps):
            """Ensure the scratchpad is always a list of BaseMessage objects."""
            messages = []
            for action, observation in intermediate_steps:
                if hasattr(action, 'log'):
                    messages.append(AIMessage(content=f"Action: {action.log}\nObservation: {observation}"))
                messages.append(HumanMessage(content=f"Observation: {observation}"))
            return messages
        
        # Get intermediate steps from previous execution (initially empty)
        intermediate_steps = getattr(self.agent, 'intermediate_steps', [])  # Fallback to empty list
        agent_scratchpad = _format_scratchpad(intermediate_steps)
        
        # **Fix: Ensure agent_scratchpad is a list of BaseMessage**
        if not isinstance(agent_scratchpad, list) or any(not isinstance(msg, BaseMessage) for msg in agent_scratchpad):
            logging.warning(f"agent_scratchpad was not a list of messages: {agent_scratchpad}")
            agent_scratchpad = _format_scratchpad([])  # Reset to an empty list

        inputs = {
            "input": query,
            "chat_history": chat_history,
            "agent_scratchpad": [agent_scratchpad]
        }
        
        logging.info(f"Agent inputs: {inputs}")
        
        try:
            response = await self.agent.ainvoke(inputs)
        except ValueError as e:
            logging.error(f"ValueError in agent execution: {str(e)}")
            logging.info(f"Agent inputs before error: {inputs}")
            # Retry with empty scratchpad if error persists
            agent_scratchpad = _format_scratchpad([])
            inputs["agent_scratchpad"] = agent_scratchpad
            response = await self.agent.ainvoke(inputs)
        
        logging.info(f"Agent response: {response}")
        
        # Ensure agent_scratchpad in response is a list
        if "agent_scratchpad" in response and (not isinstance(response["agent_scratchpad"], list) or 
                                               any(not isinstance(msg, BaseMessage) for msg in response["agent_scratchpad"])):
            logging.warning(f"agent_scratchpad was not a list of messages: {response['agent_scratchpad']}")
            response["agent_scratchpad"] = _format_scratchpad([])  # Reset it

        # Update intermediate_steps for next call (if supported by your agent)
        if hasattr(self.agent, 'intermediate_steps'):
            self.agent.intermediate_steps = response.get('intermediate_steps', [])
        
        return response["output"]
