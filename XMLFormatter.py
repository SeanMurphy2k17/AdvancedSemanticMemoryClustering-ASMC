"""
XMLFormatter - Structured data formatting for LLM cognition clarity

Converts unstructured text/dict data into clean XML for small LLMs.
Reduces cognitive load by providing semantic scaffolding.
"""

import re
from typing import Dict, List, Any, Union


class XMLFormatter:
    """
    Universal XML formatter for ASMC and GhostEngine data.
    
    Design principles:
    - Hybrid: Simple values as attributes, complex as nested tags
    - Semantic: Use meaningful tag names, not generic wrappers
    - Compact: Self-closing for empty, minimal whitespace
    - Preserve: Keep full thoughts/text, preserve newlines
    """
    
    def __init__(self):
        self.attribute_length_threshold = 50  # chars - above this, use nested tag
        
    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters"""
        if not isinstance(text, str):
            text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        return text
    
    def _escape_attribute(self, text: str) -> str:
        """Escape for attribute values (includes quote escaping)"""
        return self._escape_xml(text)
    
    def _is_simple_value(self, value: Any) -> bool:
        """Check if value is simple enough for attribute"""
        if value is None or value == "":
            return True
        if isinstance(value, (dict, list)):
            return False
        str_val = str(value)
        return len(str_val) <= self.attribute_length_threshold and '\n' not in str_val
    
    def _format_value(self, value: Any, indent: int = 0) -> str:
        """Recursively format a value as XML"""
        indent_str = "  " * indent
        
        if value is None or value == "":
            return ""
        
        if isinstance(value, dict):
            # For dicts, format as nested tags
            result = ""
            for k, v in value.items():
                tag_name = k.replace(' ', '_').replace('-', '_')
                if self._is_simple_value(v):
                    if v is None or v == "":
                        result += f"{indent_str}<{tag_name}/>\n"
                    else:
                        result += f"{indent_str}<{tag_name}>{self._escape_xml(str(v))}</{tag_name}>\n"
                else:
                    result += f"{indent_str}<{tag_name}>\n"
                    result += self._format_value(v, indent + 1)
                    result += f"{indent_str}</{tag_name}>\n"
            return result
        
        elif isinstance(value, list):
            # For lists, format each item
            result = ""
            for item in value:
                result += self._format_value(item, indent)
            return result
        
        else:
            # Simple value - just return escaped text
            return f"{indent_str}{self._escape_xml(str(value))}\n"
    
    def format(self, data: Union[Dict, List, Any], root_tag: str, 
               attributes: Dict = None, use_hybrid: bool = True) -> str:
        """
        Format data as XML with hybrid attribute/nested approach.
        
        Args:
            data: Data to format (dict, list, or primitive)
            root_tag: Root XML tag name
            attributes: Optional attributes for root tag
            use_hybrid: Use hybrid mode (simple→attributes, complex→nested)
        
        Returns:
            Well-formed XML string
        """
        attrs = attributes or {}
        
        # Build attribute string
        attr_str = ""
        nested_content = ""
        
        if isinstance(data, dict) and use_hybrid:
            # Separate simple (attributes) from complex (nested)
            for key, value in data.items():
                if self._is_simple_value(value) and key not in ['thought', 'summary', 'context']:
                    # Use as attribute
                    if value is not None and value != "":
                        attrs[key] = value
                else:
                    # Use as nested tag
                    tag_name = key.replace(' ', '_').replace('-', '_')
                    if value is None or value == "":
                        nested_content += f"  <{tag_name}/>\n"
                    elif isinstance(value, str) and '\n' in value:
                        nested_content += f"  <{tag_name}>{self._escape_xml(value)}</{tag_name}>\n"
                    else:
                        nested_content += f"  <{tag_name}>{self._escape_xml(str(value))}</{tag_name}>\n"
        else:
            # Not a dict or not hybrid - just nest everything
            nested_content = self._format_value(data, indent=1)
        
        # Build attribute string from attrs dict
        if attrs:
            attr_str = " " + " ".join([f'{k}="{self._escape_attribute(str(v))}"' for k, v in attrs.items()])
        
        # Build final XML
        if nested_content.strip():
            return f"<{root_tag}{attr_str}>\n{nested_content}</{root_tag}>"
        else:
            return f"<{root_tag}{attr_str}/>"
    
    def format_sensor_data(self, sensor_dict: Dict) -> str:
        """
        Format sensor data as structured XML.
        
        Expected dict keys:
            level, size, position, room, doors, objects, npcs, inventory, health, max_health
        """
        xml = ""
        
        # Dungeon info
        if 'level' in sensor_dict or 'size' in sensor_dict:
            dungeon_attrs = {}
            if 'level' in sensor_dict:
                dungeon_attrs['level'] = sensor_dict['level']
            if 'size' in sensor_dict:
                dungeon_attrs['size'] = sensor_dict['size']
            xml += "  " + self.format(None, 'dungeon', attributes=dungeon_attrs) + "\n"
        
        # Agent state
        agent_attrs = {}
        if 'position' in sensor_dict:
            # Convert (1.0, 1.0) to "1,1"
            pos = sensor_dict['position']
            if isinstance(pos, tuple):
                agent_attrs['position'] = f"{int(pos[0])},{int(pos[1])}"
            else:
                agent_attrs['position'] = str(pos)
        if 'room' in sensor_dict:
            agent_attrs['room'] = sensor_dict['room']
        if 'health' in sensor_dict:
            agent_attrs['health'] = sensor_dict['health']
        if 'max_health' in sensor_dict:
            agent_attrs['max_health'] = sensor_dict['max_health']
        
        if agent_attrs:
            xml += "  " + self.format(None, 'agent', attributes=agent_attrs) + "\n"
        
        # Environment state
        env_attrs = {}
        if 'doors' in sensor_dict:
            env_attrs['doors'] = sensor_dict['doors']
        if 'objects' in sensor_dict:
            env_attrs['objects'] = sensor_dict['objects'] if sensor_dict['objects'] else 'empty'
        if 'npcs' in sensor_dict:
            env_attrs['npcs'] = sensor_dict['npcs'] if sensor_dict['npcs'] else 'none'
        
        if env_attrs:
            xml += "  " + self.format(None, 'environment', attributes=env_attrs) + "\n"
        
        # Inventory
        if 'inventory' in sensor_dict:
            inv = sensor_dict['inventory']
            if inv and inv != 'empty':
                xml += f"  <inventory>{self._escape_xml(inv)}</inventory>\n"
            else:
                xml += "  <inventory/>\n"
        
        return xml.rstrip()
    
    def format_stm_memory(self, memory_entry: Dict, age: str = "unknown") -> str:
        """
        Format STM memory entry as XML.
        
        Expected keys in memory_entry:
            user_input, ai_response, thought, objective, action, result, semantic_summary, timestamp
        """
        attrs = {'age': age}
        nested = ""
        
        # Extract state from user_input (if it's sensor data)
        if 'user_input' in memory_entry:
            # Simple extraction - just store first line as context
            state_preview = memory_entry['user_input'].split('\n')[0]
            if state_preview:
                nested += f"  <state>{self._escape_xml(state_preview)}</state>\n"
        
        # Thought (preserve full text)
        if 'thought' in memory_entry and memory_entry['thought']:
            thought = memory_entry['thought']
            nested += f"  <thought>{self._escape_xml(thought)}</thought>\n"
        elif 'semantic_summary' in memory_entry:
            # Fallback to summary if thought not separated
            nested += f"  <thought>{self._escape_xml(memory_entry['semantic_summary'])}</thought>\n"
        
        # Action
        if 'action' in memory_entry and memory_entry['action']:
            nested += f"  <action>{self._escape_xml(memory_entry['action'])}</action>\n"
        
        # Result (consequence of action - CAUSAL LINK!)
        if 'result' in memory_entry and memory_entry['result']:
            nested += f"  <result>{self._escape_xml(memory_entry['result'])}</result>\n"
        
        # Full context as fallback (for backward compat during transition)
        if not nested.strip() and 'full_context' in memory_entry:
            # Old format - just wrap the whole thing
            context = memory_entry['full_context'][:200]  # Limit for safety
            nested += f"  <context>{self._escape_xml(context)}</context>\n"
        
        # Build attribute string (can't use backslash in f-string expression)
        attr_str = ' '.join([f'{k}="{self._escape_attribute(str(v))}"' for k, v in attrs.items()])
        return f"<memory {attr_str}>\n{nested}</memory>"
    
    def format_ltm_memory(self, memory_entry: Dict) -> str:
        """
        Format LTM memory entry as XML.
        
        Expected keys:
            memory_text, metadata, relevance_score
        """
        attrs = {}
        
        # Relevance score
        if 'relevance_score' in memory_entry:
            score = memory_entry['relevance_score']
            attrs['relevance'] = f"{score:.2f}"
        
        # Memory ID from metadata
        if 'metadata' in memory_entry and 'memory_id' in memory_entry['metadata']:
            attrs['id'] = memory_entry['metadata']['memory_id']
        
        nested = ""
        
        # Summary from metadata if available
        if 'metadata' in memory_entry:
            meta = memory_entry['metadata']
            if 'semantic_summary' in meta:
                nested += f"  <summary>{self._escape_xml(meta['semantic_summary'])}</summary>\n"
        
        # Full memory text as context
        if 'memory_text' in memory_entry:
            text = memory_entry['memory_text'][:300]  # Reasonable limit
            nested += f"  <context>{self._escape_xml(text)}</context>\n"
        
        # Build attribute string (can't use backslash in f-string expression)
        attr_str = ' '.join([f'{k}="{self._escape_attribute(str(v))}"' for k, v in attrs.items()])
        if nested.strip():
            return f"<memory {attr_str}>\n{nested}</memory>"
        else:
            return f"<memory {attr_str}/>"
    
    def preview_tokens(self, xml_string: str) -> int:
        """
        Rough token count estimation for debugging.
        
        Very rough heuristic: ~4 chars per token for English text
        """
        return len(xml_string) // 4
    
    def format_memory_collection(self, memories: List[Dict], 
                                 collection_type: str = "memories",
                                 formatter_func = None) -> str:
        """
        Format a list of memories as XML collection.
        
        Args:
            memories: List of memory dicts
            collection_type: Tag name for collection
            formatter_func: Function to format individual memory (defaults to format_stm_memory)
        
        Returns:
            XML string with collection wrapper
        """
        if not memories:
            return f"<{collection_type}/>"
        
        formatter = formatter_func or self.format_stm_memory
        
        xml = f"<{collection_type} count=\"{len(memories)}\">\n"
        for idx, mem in enumerate(memories):
            age = f"{len(memories) - idx}_cycles_ago" if collection_type == "recent_memories" else None
            if age:
                xml += formatter(mem, age=age) + "\n"
            else:
                xml += formatter(mem) + "\n"
        xml += f"</{collection_type}>"
        
        return xml
    
    def format_full_cognition_context(self, sensor_dict: Dict, 
                                      stm_memories: List[Dict] = None,
                                      ltm_memories: List[Dict] = None,
                                      current_objective: str = "",
                                      previous_thoughts: str = "",
                                      last_action_result: str = "") -> str:
        """
        Format complete cognition context as single unified XML document.
        
        This creates one coherent XML tree instead of concatenated fragments.
        Easier for small LLMs to parse and understand.
        
        Args:
            sensor_dict: Current sensor data as dict
            stm_memories: Short-term memory entries
            ltm_memories: Long-term memory entries
            current_objective: Agent's current objective
            previous_thoughts: Previous thought accumulation
        
        Returns:
            Complete XML cognition context document
        """
        xml = "<cognition_context>\n"
        
        # Previous action feedback (immediate consequence visibility)
        if last_action_result and last_action_result.strip():
            xml += f"  <previous_action_feedback>{self._escape_xml(last_action_result)}</previous_action_feedback>\n\n"
        
        # Current objective (if set)
        if current_objective and current_objective.strip():
            xml += f"  <current_objective>{self._escape_xml(current_objective)}</current_objective>\n\n"
        
        # Current state (sensor data)
        xml += "  <current_state>\n"
        sensor_xml = self.format_sensor_data(sensor_dict)
        # Indent sensor XML
        for line in sensor_xml.split('\n'):
            if line.strip():
                xml += f"    {line}\n"
        xml += "  </current_state>\n"
        
        # Recent memories (STM)
        if stm_memories:
            xml += f"\n  <recent_memories count=\"{len(stm_memories)}\">\n"
            for idx, mem in enumerate(stm_memories):
                age = f"{len(stm_memories) - idx}_cycles_ago"
                mem_xml = self.format_stm_memory(mem, age=age)
                # Indent memory XML
                for line in mem_xml.split('\n'):
                    if line.strip():
                        xml += f"    {line}\n"
            xml += "  </recent_memories>\n"
        
        # Long-term memories (LTM)
        if ltm_memories:
            xml += f"\n  <long_term_memories session=\"previous\" count=\"{len(ltm_memories)}\">\n"
            for mem in ltm_memories:
                mem_xml = self.format_ltm_memory(mem)
                # Indent memory XML
                for line in mem_xml.split('\n'):
                    if line.strip():
                        xml += f"    {line}\n"
            xml += "  </long_term_memories>\n"
        
        # Previous thoughts (if any)
        if previous_thoughts and previous_thoughts.strip():
            xml += f"\n  <previous_cycles>\n"
            xml += f"    <history>{self._escape_xml(previous_thoughts[:500])}</history>\n"
            xml += "  </previous_cycles>\n"
        
        xml += "</cognition_context>"
        
        return xml
    
    def format_tools(self, tools_list: List[str], tool_descriptions: Dict[str, str]) -> str:
        """
        Format tool list as XML for action selection.
        
        Args:
            tools_list: List of available tool names
            tool_descriptions: Dict mapping tool names to descriptions
        
        Returns:
            XML tools block
        """
        if not tools_list:
            return "<tools/>"
        
        xml = "<tools>\n"
        for tool in tools_list:
            desc = tool_descriptions.get(tool, tool)
            xml += f'  <tool name="{tool}" desc="{desc}"/>\n'
        xml += "</tools>"
        
        return xml
    
    def format_action_categories(self, action_categories: Dict[str, Dict[str, str]]) -> str:
        """
        Format categorized actions as XML structure for frame-based action system.
        
        Args:
            action_categories: {"perception": {action_name: desc}, "movement": {action_name: desc}, ...}
        
        Returns:
            XML with categorized actions
        """
        if not action_categories:
            return "<action_categories/>"
        
        xml = "<action_categories>\n"
        
        for category_name, actions in action_categories.items():
            xml += f'  <{category_name}>\n'
            for action_name, action_desc in actions.items():
                xml += f'    <action name="{action_name}">{self._escape_xml(action_desc)}</action>\n'
            xml += f'  </{category_name}>\n'
        
        xml += "</action_categories>"
        return xml
    
    def format_objective_extraction(self, thought: str, instruction: str) -> str:
        """
        Format objective extraction prompt as XML.
        
        Args:
            thought: The thought output to extract objective from
            instruction: Instruction for extraction
        
        Returns:
            XML objective extraction document
        """
        xml = "<objective_extraction>\n"
        xml += f"  <thoughts>{self._escape_xml(thought)}</thoughts>\n"
        xml += f"  <instruction>{self._escape_xml(instruction)}</instruction>\n"
        xml += "</objective_extraction>"
        
        return xml

