"""
Base Agent Class - Foundation for all specialized agents
Handles MCP tool discovery and common functionality
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

from src.utils.mcp_registry import get_mcp_registry
from src.utils.memory import MemoryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str, role: str, memory_store: Optional[MemoryStore] = None):
        self.name = name
        self.role = role
        self.memory_store = memory_store
        self.mcp_registry = get_mcp_registry()
        self.available_tools = self._discover_tools()
    
    def _discover_tools(self) -> List[str]:
        """Dynamically discover available tools for this agent's role."""
        tools = self.mcp_registry.query_by_agent_role(self.role)
        logger.info(f"{self.name} discovered tools: {tools}")
        return tools
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema information for a tool."""
        return self.mcp_registry.get_tool_schema(tool_name)
    
    def call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Call an MCP tool discovered at runtime.
        This enforces dynamic tool discovery - no hardcoded APIs.
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool {tool_name} not available for {self.role}")
        
        tool = self.mcp_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found in registry")
        
        if not tool.impl:
            raise RuntimeError(f"Tool {tool_name} implementation not bound")
        
        logger.info(f"{self.name} calling tool: {tool_name} with params: {params}")
        result = tool.impl(**params)
        return result
    
    def commit_to_memory(self, entry_type: str, content: Dict[str, Any]) -> str:
        """Store information in shared memory through MCP tool when available."""
        if "commit_memory" in self.available_tools:
            try:
                result = self.call_mcp_tool("commit_memory", {
                    "entry_type": entry_type,
                    "content": content
                })
                entry_id = result.get("entry_id", "") if isinstance(result, dict) else ""
                logger.info(f"{self.name} committed to memory via MCP: {entry_id}")
                return entry_id
            except Exception as e:
                logger.warning(f"{self.name} MCP memory commit failed, falling back to direct store: {e}")

        if not self.memory_store:
            logger.warning(f"{self.name} has no memory store")
            return ""

        entry_id = self.memory_store.commit(entry_type, content)
        logger.info(f"{self.name} committed to memory (fallback): {entry_id}")
        return entry_id
    
    def query_memory(self, entry_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query shared memory for context."""
        if not self.memory_store:
            return []
        
        return self.memory_store.query(entry_type=entry_type, limit=limit)
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main task. Must be implemented by subclasses."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, role={self.role})"
