"""
Researcher Agent: Deep Web Research for A-MEM

Implements JIT (Just-in-Time) research: When retrieval confidence is low,
this agent performs deep web research to find relevant information and
creates new atomic notes from the findings.

Uses GetWeb MCP (felo-search, jina-reader) and Firecrawl for content extraction.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from ..models.note import AtomicNote
from ..utils.llm import LLMService
from ..config import settings


class ResearcherAgent:
    """
    Researcher Agent for deep web research.
    
    Workflow:
    1. Receive query (from low-confidence retrieval)
    2. Search web using felo-search (technical research)
    3. Extract top URLs with jina-reader (high-quality content)
    4. Summarize findings with LLM
    5. Return as AtomicNote (ready to be stored)
    """
    
    def __init__(
        self, 
        llm_service: Optional[LLMService] = None, 
        max_sources: int = 1,
        mcp_tool_callback: Optional[Callable[[str, dict], Any]] = None
    ):
        """
        Initialize Researcher Agent.
        
        Args:
            llm_service: LLM service for note creation
            max_sources: Maximum number of sources to extract
            mcp_tool_callback: Optional callback function to call MCP tools.
                             If provided, will be used instead of HTTP fallbacks.
                             Signature: async def mcp_tool_callback(tool_name: str, arguments: dict) -> Any
        """
        self.llm = llm_service or LLMService()
        self.max_sources = max_sources  # Limit sources for efficiency (configurable, default: 1)
        self.max_content_length = 10000  # Max chars per source
        self.mcp_tool_callback = mcp_tool_callback  # Optional MCP tool callback
    
    async def research(self, query: str, context: Optional[str] = None) -> List[AtomicNote]:
        """
        Performs deep web research on a query.
        
        Args:
            query: The research query
            context: Optional context about why this research is needed
            
        Returns:
            List of AtomicNote objects with research findings
        """
        print(f"[RESEARCHER] Starting research for: {query}")
        
        # Step 1: Search web (using felo-search for technical content)
        search_results = await self._search_web(query)
        
        if not search_results:
            print("[RESEARCHER] No search results found")
            return []
        
        # Step 2: Extract content from top URLs
        extracted_content = await self._extract_content(search_results[:self.max_sources])
        
        if not extracted_content:
            print("[RESEARCHER] No content extracted")
            return []
        
        # Step 3: Summarize and create notes
        notes = await self._create_notes_from_research(query, extracted_content, context)
        
        print(f"[RESEARCHER] Research complete: {len(notes)} notes created")
        return notes
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Searches the web using MCP tools (if available) or HTTP-based fallbacks.
        
        Strategy:
        1. Try MCP tools (felo-search, duckduckgo-search) if callback available
        2. Fallback to Google Search API (if configured)
        3. Fallback to DuckDuckGo HTTP search
        
        Returns list of search results with URLs and snippets.
        """
        print(f"[RESEARCHER] Searching web for: {query}")
        
        # Strategy 1: Try MCP tools if callback available
        if self.mcp_tool_callback:
            try:
                # Try felo-search first (best for technical queries)
                try:
                    mcp_result = await self.mcp_tool_callback("mcp_getweb_felo-search", {"query": query})
                    if mcp_result:
                        # Parse MCP result (format depends on MCP tool)
                        results = self._parse_mcp_search_result(mcp_result)
                        if results:
                            print(f"[RESEARCHER] Found {len(results)} results via MCP felo-search")
                            return results
                except Exception as e:
                    print(f"[RESEARCHER] MCP felo-search failed: {e}, trying duckduckgo-search...")
                
                # Try duckduckgo-search as fallback
                try:
                    mcp_result = await self.mcp_tool_callback("mcp_getweb_duckduckgo-search", {"query": query, "numResults": self.max_sources})
                    if mcp_result:
                        results = self._parse_mcp_search_result(mcp_result)
                        if results:
                            print(f"[RESEARCHER] Found {len(results)} results via MCP duckduckgo-search")
                            return results
                except Exception as e:
                    print(f"[RESEARCHER] MCP duckduckgo-search failed: {e}, trying HTTP fallbacks...")
            except Exception as e:
                print(f"[RESEARCHER] MCP tool callback error: {e}, falling back to HTTP...")
        
        # Strategy 2: Try Google Search API (if configured)
        try:
            from .researcher_tools import search_google
            
            results = await asyncio.to_thread(search_google, query, max_results=self.max_sources)
            
            if results:
                print(f"[RESEARCHER] Found {len(results)} results via Google Search API")
                return results
        except Exception as e:
            print(f"[RESEARCHER] Google Search failed: {e}, trying DuckDuckGo...")
        
        # Strategy 3: Fallback to DuckDuckGo HTTP search
        try:
            from .researcher_tools import search_duckduckgo
            
            results = await asyncio.to_thread(search_duckduckgo, query, max_results=self.max_sources)
            
            if results:
                print(f"[RESEARCHER] Found {len(results)} results via DuckDuckGo (HTTP fallback)")
                return results
            else:
                print("[RESEARCHER] No search results found")
                return []
                
        except ImportError:
            print("[RESEARCHER] researcher_tools not available")
            return []
        except Exception as e:
            print(f"[RESEARCHER] HTTP search error: {e}")
            return []
    
    def _parse_mcp_search_result(self, mcp_result: Any) -> List[Dict[str, Any]]:
        """
        Parses MCP tool result into standardized format.
        
        Handles different MCP tool response formats.
        """
        results = []
        
        # Handle different MCP result formats
        if isinstance(mcp_result, list):
            for item in mcp_result:
                if isinstance(item, dict):
                    results.append({
                        "url": item.get("url") or item.get("link") or item.get("href", ""),
                        "title": item.get("title") or item.get("text", ""),
                        "snippet": item.get("snippet") or item.get("description", "")
                    })
        elif isinstance(mcp_result, dict):
            # Single result or wrapped result
            if "url" in mcp_result or "link" in mcp_result:
                results.append({
                    "url": mcp_result.get("url") or mcp_result.get("link", ""),
                    "title": mcp_result.get("title") or mcp_result.get("text", ""),
                    "snippet": mcp_result.get("snippet") or mcp_result.get("description", "")
                })
            elif "results" in mcp_result:
                # Wrapped results
                for item in mcp_result.get("results", []):
                    if isinstance(item, dict):
                        results.append({
                            "url": item.get("url") or item.get("link", ""),
                            "title": item.get("title") or item.get("text", ""),
                            "snippet": item.get("snippet") or item.get("description", "")
                        })
        
        return results[:self.max_sources]
    
    async def _extract_content(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts content from URLs using MCP tools (if available) or HTTP-based fallbacks.
        
        Strategy:
        1. Try MCP jina-reader if callback available
        2. Fallback to HTTP-based Jina Reader (local Docker or cloud)
        3. Fallback to Unstructured for PDFs
        4. Fallback to readability
        
        Returns list of extracted content with URLs.
        """
        extracted = []
        
        for result in search_results:
            url = result.get("url") or result.get("link")
            if not url:
                continue
            
            try:
                print(f"[RESEARCHER] Extracting content from: {url}")
                
                # Strategy 1: Try MCP jina-reader if callback available
                if self.mcp_tool_callback:
                    try:
                        mcp_result = await self.mcp_tool_callback("mcp_getweb_jina-reader", {"url": url})
                        if mcp_result:
                            content = self._parse_mcp_content_result(mcp_result)
                            if content:
                                extracted.append({
                                    "url": url,
                                    "content": content[:self.max_content_length],
                                    "title": result.get("title", ""),
                                    "snippet": result.get("snippet", "")
                                })
                                print(f"[RESEARCHER] Successfully extracted {len(content)} chars from {url} (MCP jina-reader)")
                                continue
                    except Exception as e:
                        print(f"[RESEARCHER] MCP jina-reader failed: {e}, trying HTTP fallbacks...")
                
                # Strategy 2: HTTP-based extraction
                from .researcher_tools import (
                    extract_with_jina_reader,
                    extract_pdf_with_unstructured,
                    extract_with_readability
                )
                
                # Check if URL is a PDF - use Unstructured for PDFs
                if url.lower().endswith('.pdf') or 'pdf' in url.lower():
                    if settings.UNSTRUCTURED_ENABLED:
                        content = await asyncio.to_thread(
                            extract_pdf_with_unstructured, 
                            url, 
                            self.max_content_length
                        )
                        if content:
                            extracted.append({
                                "url": url,
                                "content": content,
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", "")
                            })
                            print(f"[RESEARCHER] Successfully extracted {len(content)} chars from PDF {url}")
                            continue
                
                # Try Jina Reader first (best quality for web content)
                content = await asyncio.to_thread(
                    extract_with_jina_reader, 
                    url, 
                    self.max_content_length
                )
                
                # Fallback to readability if Jina fails
                if not content:
                    print(f"[RESEARCHER] Jina Reader failed, trying fallback for {url}")
                    content = await asyncio.to_thread(extract_with_readability, url)
                
                if content:
                    extracted.append({
                        "url": url,
                        "content": content,
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", "")
                    })
                    print(f"[RESEARCHER] Successfully extracted {len(content)} chars from {url}")
                else:
                    print(f"[RESEARCHER] Failed to extract content from {url}")
                
            except ImportError:
                print(f"[RESEARCHER] researcher_tools not available - skipping {url}")
                continue
            except Exception as e:
                print(f"[RESEARCHER] Error extracting from {url}: {e}")
                continue
        
        return extracted
    
    def _parse_mcp_content_result(self, mcp_result: Any) -> Optional[str]:
        """
        Parses MCP tool content result into text string.
        
        Handles different MCP tool response formats.
        """
        if isinstance(mcp_result, str):
            return mcp_result
        elif isinstance(mcp_result, dict):
            # Try common content fields
            return (
                mcp_result.get("content") or 
                mcp_result.get("text") or 
                mcp_result.get("markdown") or
                mcp_result.get("body")
            )
        elif isinstance(mcp_result, list):
            # Join list items
            return "\n".join(str(item) for item in mcp_result)
        return None
    
    async def _create_notes_from_research(
        self,
        query: str,
        extracted_content: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> List[AtomicNote]:
        """
        Creates AtomicNote objects from extracted research content.
        
        Uses LLM to summarize and extract key information.
        """
        notes = []
        
        for i, content_item in enumerate(extracted_content):
            url = content_item.get("url", "")
            text = content_item.get("content", "")[:self.max_content_length]
            
            if not text:
                continue
            
            # Use LLM to create a structured note from the content
            prompt = f"""Create a concise, atomic note from this research content.

Query: {query}
Source URL: {url}
Context: {context or "General research"}

Content:
{text}

Return a JSON object with:
- "summary": Brief contextual summary (max 100 chars)
- "keywords": List of 3-5 key terms
- "tags": List of 2-3 category tags
- "type": One of: "rule", "procedure", "concept", "tool", "reference", "integration"
- "note_content": The main note content (concise, atomic, max 500 words)

Focus on extracting actionable insights, not just summarizing.
Return ONLY the JSON object, no markdown formatting."""

            try:
                response = self.llm._call_llm(prompt)
                data = self.llm._clean_json_response(response)
                
                # Create AtomicNote
                note = AtomicNote(
                    content=data.get("note_content", text[:500]),
                    contextual_summary=data.get("summary", ""),
                    keywords=data.get("keywords", []),
                    tags=data.get("tags", []),
                    type=data.get("type", "reference"),
                    metadata={
                        "source": "researcher_agent",
                        "source_url": url,
                        "research_query": query,
                        "context": context
                    }
                )
                
                notes.append(note)
                print(f"[RESEARCHER] Created note: {note.id[:8]} - {note.contextual_summary[:50]}")
                
            except Exception as e:
                print(f"[RESEARCHER] Error creating note from content: {e}")
                continue
        
        return notes
    
    def research_sync(self, query: str, context: Optional[str] = None) -> List[AtomicNote]:
        """
        Synchronous wrapper for research().
        Useful for testing or non-async contexts.
        """
        return asyncio.run(self.research(query, context))

