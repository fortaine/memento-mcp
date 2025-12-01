"""
LLM Service: Embedding, Metadata Extraction, Link Checking, Memory Evolution

Supports Ollama (local) and OpenRouter (cloud) for LLM and embeddings.
"""

import json
import sys
import requests
from typing import List, Tuple, Dict, Any, Optional
from ..models.note import AtomicNote, NoteRelation
from ..config import settings

class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        
        # Ollama Settings
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.ollama_llm_model = settings.OLLAMA_LLM_MODEL
        self.ollama_embedding_model = settings.OLLAMA_EMBEDDING_MODEL
        
        # OpenRouter Settings
        self.openrouter_api_key = settings.OPENROUTER_API_KEY
        self.openrouter_base_url = settings.OPENROUTER_BASE_URL
        self.openrouter_llm_model = settings.OPENROUTER_LLM_MODEL
        self.openrouter_embedding_model = settings.OPENROUTER_EMBEDDING_MODEL
        
        # Compatibility with old code
        self.llm_model = settings.LLM_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL
        
    def _call_ollama(self, prompt: str, system: Optional[str] = None) -> str:
        """Calls Ollama LLM."""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.ollama_llm_model,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"Ollama LLM Error: {e}", file=sys.stderr)
            raise
    
    def _call_openrouter(self, prompt: str, system: Optional[str] = None) -> str:
        """Calls OpenRouter LLM."""
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment variables")
        
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/tobsentobsentobs/a-mem-mcp-server",  # Optional
                "X-Title": "A-MEM Agentic Memory System"  # Optional
            }
            
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.openrouter_llm_model,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenRouter LLM Error: {e}", file=sys.stderr)
            raise
    
    def _call_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """Calls LLM (provider-dependent)."""
        if self.provider == "openrouter":
            try:
                return self._call_openrouter(prompt, system)
            except Exception as e:
                print(f"OpenRouter failed, falling back to Ollama: {e}", file=sys.stderr)
                # Fallback to Ollama
                return self._call_ollama(prompt, system)
        else:
            return self._call_ollama(prompt, system)

    def _get_embedding_ollama(self, text: str) -> List[float]:
        """Calls Ollama Embedding API."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.ollama_embedding_model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Ollama Embedding Error: {e}", file=sys.stderr)
            # Fallback: Mock embedding
            return [0.0] * 768  # nomic-embed-text has 768 dimensions
    
    def _get_embedding_openrouter(self, text: str) -> List[float]:
        """Calls OpenRouter Embedding API."""
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment variables")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/tobsentobsentobs/a-mem-mcp-server",
                "X-Title": "A-MEM Agentic Memory System"
            }
            
            response = requests.post(
                f"{self.openrouter_base_url}/embeddings",
                headers=headers,
                json={
                    "model": self.openrouter_embedding_model,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"OpenRouter Embedding Error: {e}", file=sys.stderr)
            raise

    def _clean_json_response(self, content: str) -> Dict[str, Any]:
        """Entfernt Markdown Fences und parst JSON."""
        content = content.strip()
        if content.startswith("```"):
            # Entferne erste Zeile (```json) und letzte Zeile (```)
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Versuche JSON aus Text zu extrahieren
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Calculates embedding (provider-dependent)."""
        if self.provider == "openrouter":
            try:
                return self._get_embedding_openrouter(text)
            except Exception as e:
                print(f"OpenRouter embedding failed, falling back to Ollama: {e}", file=sys.stderr)
                # Fallback to Ollama
                return self._get_embedding_ollama(text)
        else:
            return self._get_embedding_ollama(text)

    def extract_metadata(self, content: str) -> dict:
        """Extracts metadata (Summary, Keywords, Tags, Type) via LLM (provider-dependent)."""
        prompt = f"""Analyze this memory fragment: "{content}"

Return a valid JSON object with these exact keys:
- "summary": A short contextual summary (max 100 characters)
- "keywords": A list of 3-5 key terms
- "tags": A list of 2-3 category tags
- "type": One of: "rule", "procedure", "concept", "tool", "reference", "integration"
  * "rule": Contains imperative instructions ("Never X", "Always Y")
  * "procedure": Contains numbered steps or sequential instructions
  * "concept": Explains something, no commands
  * "tool": Describes functions, APIs, or utilities
  * "reference": Contains tables, comparison lists, cheatsheets
  * "integration": Describes connections between systems

Example format:
{{"summary": "Brief summary here", "keywords": ["term1", "term2"], "tags": ["category1", "category2"], "type": "concept"}}

Return ONLY the JSON object, no markdown formatting, no explanations."""
        
        try:
            response_text = self._call_llm(prompt)
            data = self._clean_json_response(response_text)
            # Validate type
            valid_types = {"rule", "procedure", "concept", "tool", "reference", "integration"}
            if "type" in data and data["type"] not in valid_types:
                data["type"] = "concept"  # Default fallback
            return data
        except Exception as e:
            print(f"LLM Extraction Error: {e}", file=sys.stderr)
            return {"summary": content[:50]+"...", "keywords": [], "tags": [], "type": "concept"}

    def check_link(self, note_a: AtomicNote, note_b: AtomicNote) -> Tuple[bool, Optional[NoteRelation]]:
        """Checks if two notes should be linked."""
        prompt = f"""Node A: {note_a.content} (Summary: {note_a.contextual_summary})
Node B: {note_b.content} (Summary: {note_b.contextual_summary})

Are these fundamentally related? 

Return a valid JSON object with these exact keys:
- "related": boolean (true if related, false otherwise)
- "type": string (one of: "extends", "contradicts", "supports", "relates_to")
- "reasoning": string (brief explanation)

Example format:
{{"related": true, "type": "relates_to", "reasoning": "Both discuss similar concepts"}}

Return ONLY the JSON object, no markdown formatting, no explanations."""
        
        try:
            response_text = self._call_llm(prompt)
            data = self._clean_json_response(response_text)
            
            if data.get("related"):
                relation = NoteRelation(
                    source_id=note_a.id,
                    target_id=note_b.id,
                    relation_type=data.get("type", "relates_to"),
                    reasoning=data.get("reasoning", "Semantic similarity")
                )
                return True, relation
        except Exception as e:
            print(f"LLM Linking Error: {e}", file=sys.stderr)
        
        return False, None

    def evolve_memory(self, new_note: AtomicNote, existing_note: AtomicNote) -> Optional[AtomicNote]:
        """
        Memory Evolution (Paper Section 3.3): 
        Checks if existing_note should be updated based on new_note.
        Formula: mj* = LLM(mn || M_near || mj || Ps3)
        """
        prompt = f"""New Memory (mn): 
Content: {new_note.content}
Summary: {new_note.contextual_summary}
Keywords: {', '.join(new_note.keywords)}
Tags: {', '.join(new_note.tags)}

Existing Memory (mj):
Content: {existing_note.content}
Summary: {existing_note.contextual_summary}
Keywords: {', '.join(existing_note.keywords)}
Tags: {', '.join(existing_note.tags)}

Based on the new memory, should the existing memory be updated or refined?
Consider: Does the new information add context, correct errors, extend concepts, or reveal new patterns?

Return a valid JSON object with these exact keys:
- "should_update": boolean
- "updated_summary": string (original summary if should_update is false)
- "updated_keywords": list of strings (original keywords if should_update is false)
- "updated_tags": list of strings (original tags if should_update is false)
- "reasoning": string (explanation)

If should_update is false, return the original values for summary, keywords, and tags.

Return ONLY the JSON object, no markdown formatting, no explanations."""
        
        try:
            response_text = self._call_llm(prompt)
            data = self._clean_json_response(response_text)
            
            if data.get("should_update", False):
                # Create updated note with new information
                evolved_note = AtomicNote(
                    id=existing_note.id,  # ID stays the same
                    content=existing_note.content,  # Content stays the same (only metadata is refined)
                    contextual_summary=data.get("updated_summary", existing_note.contextual_summary),
                    keywords=data.get("updated_keywords", existing_note.keywords),
                    tags=data.get("updated_tags", existing_note.tags),
                    created_at=existing_note.created_at,  # Original timestamp beibehalten
                    type=existing_note.type,  # Type stays the same
                    metadata=existing_note.metadata  # Metadata stays the same
                )
                return evolved_note
        except Exception as e:
            print(f"LLM Evolution Error: {e}", file=sys.stderr)
        
        return None
