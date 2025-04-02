import json
from datetime import datetime

def format_response(result):
    """Format API responses for consistency."""
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str, indent=2)
    return str(result)

def log_query(query, response, chat_history=None):
    """Log queries and responses for analytics."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "response": response,
        "chat_history_length": len(chat_history) if chat_history else 0
    }
    # This could write to a database or log file
    print(f"Query log: {json.dumps(log_entry)}")
    return log_entry