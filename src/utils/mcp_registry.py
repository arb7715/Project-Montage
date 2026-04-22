"""
MCP Tool Registry - Dynamic tool discovery without hardcoding APIs.
All agents query this registry at runtime to find available tools.
"""
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    agent_roles: List[str]  # Which agents can use this tool
    impl: Optional[Callable] = None  # Runtime implementation

class MCPRegistry:
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register core MCP tools available at runtime."""
        
        # Scriptwriter tools
        self.register(MCPTool(
            name="generate_script_segment",
            description="Generate a structured script segment (scene) from an abstract prompt",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "scene_number": {"type": "integer"},
                    "max_characters": {"type": "integer", "default": 5}
                },
                "required": ["prompt"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "scene_id": {"type": "integer"},
                    "heading": {"type": "string"},
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "dialogues": {"type": "array"},
                    "visual_cues": {"type": "array", "items": {"type": "string"}}
                }
            },
            agent_roles=["scriptwriter"],
            impl=None  # Will be bound at runtime
        ))

        # Validator tools
        self.register(MCPTool(
            name="validate_script_structure",
            description="Validate screenplay structure (scene headers, dialogue labels, actions)",
            input_schema={
                "type": "object",
                "properties": {
                    "script_text": {"type": "string"}
                },
                "required": ["script_text"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "is_valid": {"type": "boolean"},
                    "errors": {"type": "array", "items": {"type": "string"}},
                    "warnings": {"type": "array", "items": {"type": "string"}}
                }
            },
            agent_roles=["validator"],
            impl=None
        ))

        # Memory tools
        self.register(MCPTool(
            name="commit_memory",
            description="Store script history, character metadata, or image references in persistent memory",
            input_schema={
                "type": "object",
                "properties": {
                    "entry_type": {"type": "string", "enum": ["script_history", "character_metadata", "image_reference"]},
                    "content": {"type": "object"}
                },
                "required": ["entry_type", "content"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "entry_id": {"type": "string"}
                }
            },
            agent_roles=["scriptwriter", "character_designer", "image_synthesizer"],
            impl=None
        ))

        # Character tools
        self.register(MCPTool(
            name="extract_character_profile",
            description="Extract and formalize character identity from script",
            input_schema={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string"},
                    "script_context": {"type": "string"}
                },
                "required": ["character_name", "script_context"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "personality_traits": {"type": "array", "items": {"type": "string"}},
                    "appearance_description": {"type": "string"},
                    "reference_style": {"type": "string"}
                }
            },
            agent_roles=["character_designer"],
            impl=None
        ))

        # Image synthesis tools
        self.register(MCPTool(
            name="generate_character_image",
            description="Generate visual representation of character",
            input_schema={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string"},
                    "appearance_description": {"type": "string"},
                    "style": {"type": "string"}
                },
                "required": ["character_name", "appearance_description"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "character_name": {"type": "string"},
                    "generated_at": {"type": "string"}
                }
            },
            agent_roles=["image_synthesizer"],
            impl=None
        ))

        # ── Phase 2 Tools ──────────────────────────────────────────────────────

        # Scene Parser tools
        self.register(MCPTool(
            name="get_task_graph",
            description="Decompose scene_manifest.json into a parallel task graph of SceneTask units",
            input_schema={
                "type": "object",
                "properties": {
                    "scene_manifest_path": {"type": "string"},
                    "output_dir": {"type": "string"}
                },
                "required": ["scene_manifest_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array"},
                    "total_scenes": {"type": "integer"},
                    "task_graph_path": {"type": "string"}
                }
            },
            agent_roles=["scene_parser"],
            impl=None
        ))

        # Voice synthesis tools
        self.register(MCPTool(
            name="voice_cloning_synthesizer",
            description="Synthesize character speech from dialogue text using TTS or voice cloning",
            input_schema={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string"},
                    "dialogue_text": {"type": "string"},
                    "output_path": {"type": "string"},
                    "voice_style": {"type": "string", "default": "neutral"}
                },
                "required": ["character_name", "dialogue_text", "output_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string"},
                    "duration_seconds": {"type": "number"},
                    "success": {"type": "boolean"}
                }
            },
            agent_roles=["voice_synthesizer"],
            impl=None
        ))

        # Video generation tools
        self.register(MCPTool(
            name="query_stock_footage",
            description="Retrieve or generate visual footage for a scene based on descriptions and character images",
            input_schema={
                "type": "object",
                "properties": {
                    "scene_id": {"type": "integer"},
                    "visual_cues": {"type": "array", "items": {"type": "string"}},
                    "character_names": {"type": "array", "items": {"type": "string"}},
                    "output_dir": {"type": "string"}
                },
                "required": ["scene_id", "visual_cues"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "frame_dir": {"type": "string"},
                    "frame_count": {"type": "integer"},
                    "raw_video_path": {"type": "string"}
                }
            },
            agent_roles=["video_generator"],
            impl=None
        ))

        # Face swap tools
        self.register(MCPTool(
            name="identity_validator",
            description="Validate character identity confidence before face mapping is applied",
            input_schema={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string"},
                    "reference_image_path": {"type": "string"},
                    "frame_path": {"type": "string"}
                },
                "required": ["character_name", "reference_image_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "identity_confirmed": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "character_name": {"type": "string"}
                }
            },
            agent_roles=["face_swap"],
            impl=None
        ))

        self.register(MCPTool(
            name="face_swapper",
            description="Blend character reference face onto video frame region",
            input_schema={
                "type": "object",
                "properties": {
                    "frame_path": {"type": "string"},
                    "reference_image_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "blend_alpha": {"type": "number", "default": 0.75}
                },
                "required": ["frame_path", "reference_image_path", "output_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_path": {"type": "string"},
                    "success": {"type": "boolean"}
                }
            },
            agent_roles=["face_swap"],
            impl=None
        ))

        # Lip sync tools
        self.register(MCPTool(
            name="lip_sync_aligner",
            description="Merge audio waveform onto video with temporal frame alignment for lip sync",
            input_schema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string"},
                    "audio_path": {"type": "string"},
                    "output_path": {"type": "string"}
                },
                "required": ["video_path", "audio_path", "output_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_path": {"type": "string"},
                    "duration_seconds": {"type": "number"},
                    "success": {"type": "boolean"}
                }
            },
            agent_roles=["lip_sync"],
            impl=None
        ))

    def register(self, tool: MCPTool):
        """Register a new MCP tool."""
        self.tools[tool.name] = tool

    def query_by_agent_role(self, role: str) -> List[str]:
        """Query available tools for a specific agent role."""
        return [name for name, tool in self.tools.items() if role in tool.agent_roles]

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_tool_schema(self, name: str) -> Dict[str, Any]:
        """Get the schema for a specific tool."""
        tool = self.get_tool(name)
        if not tool:
            return {}
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema
        }

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [asdict(tool) for tool in self.tools.values()]

    def bind_implementation(self, tool_name: str, impl: Callable):
        """Bind a runtime implementation to a tool."""
        if tool_name in self.tools:
            self.tools[tool_name].impl = impl

    def export_registry(self, filepath: str):
        """Export registry metadata as JSON (for logging/debugging)."""
        registry_meta = {
            "exported_at": datetime.now().isoformat(),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "agent_roles": tool.agent_roles,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema
                }
                for tool in self.tools.values()
            ]
        }
        with open(filepath, "w") as f:
            json.dump(registry_meta, f, indent=2)


# Global registry instance
_mcp_registry = None

def get_mcp_registry() -> MCPRegistry:
    global _mcp_registry
    if _mcp_registry is None:
        _mcp_registry = MCPRegistry()
    return _mcp_registry
