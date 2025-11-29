"""
Researcher Tools: Lightweight HTTP-based tools for web search and content extraction.

These tools are independent of MCP and can be used directly by the Researcher Agent.
"""

import requests
from typing import List, Dict, Any, Optional
import json
import re


def search_google(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches Google using Custom Search API.
    
    Requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in config.
    
    Returns list of search results with URLs and snippets.
    """
    from ..config import settings
    
    if not settings.GOOGLE_SEARCH_ENABLED or not settings.GOOGLE_API_KEY or not settings.GOOGLE_SEARCH_ENGINE_ID:
        return []
    
    try:
        import requests
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": settings.GOOGLE_API_KEY,
            "cx": settings.GOOGLE_SEARCH_ENGINE_ID,
            "q": query,
            "num": min(max_results, 10)  # Google API max is 10 per request
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append({
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "")
            })
        
        return results
        
    except Exception as e:
        print(f"[RESEARCHER] Google Search error: {e}")
        return []


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches DuckDuckGo using their instant answer API.
    
    Returns list of search results with URLs and snippets.
    """
    try:
        # DuckDuckGo Instant Answer API (no API key required)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Extract Abstract (if available)
        if data.get("AbstractText"):
            results.append({
                "url": data.get("AbstractURL", ""),
                "title": data.get("Heading", query),
                "snippet": data.get("AbstractText", "")[:200]
            })
        
        # Extract Related Topics
        for topic in data.get("RelatedTopics", [])[:max_results-1]:
            if isinstance(topic, dict) and "FirstURL" in topic:
                results.append({
                    "url": topic.get("FirstURL", ""),
                    "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                    "snippet": topic.get("Text", "")[:200]
                })
        
        # If no results, try HTML scraping (fallback)
        if not results:
            try:
                # DuckDuckGo HTML search (more reliable than API for general queries)
                search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5"
                }
                html_response = requests.get(search_url, headers=headers, timeout=15)
                html_response.raise_for_status()
                
                # Improved regex patterns for DuckDuckGo HTML results
                # DuckDuckGo uses different class names, try multiple patterns
                url_patterns = [
                    r'<a class="result__a"[^>]+href="([^"]+)"',
                    r'href="([^"]+)"[^>]*class="result__a"',
                    r'<a[^>]+class="result__url"[^>]+href="([^"]+)"',
                ]
                
                title_patterns = [
                    r'<a class="result__a"[^>]*>([^<]+)</a>',
                    r'class="result__title"[^>]*>([^<]+)</a>',
                ]
                
                snippet_patterns = [
                    r'<a class="result__snippet"[^>]*>([^<]+)</a>',
                    r'class="result__snippet"[^>]*>([^<]+)</a>',
                ]
                
                urls = []
                titles = []
                snippets = []
                
                # Try to extract with first pattern
                for pattern in url_patterns:
                    urls = re.findall(pattern, html_response.text)
                    if urls:
                        break
                
                for pattern in title_patterns:
                    titles = re.findall(pattern, html_response.text)
                    if titles:
                        break
                
                for pattern in snippet_patterns:
                    snippets = re.findall(pattern, html_response.text)
                    if snippets:
                        break
                
                # If still no results, try BeautifulSoup (if available)
                if not urls:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html_response.text, 'html.parser')
                        
                        # Find result links
                        for link in soup.find_all('a', class_=re.compile('result__a|result__url')):
                            href = link.get('href', '')
                            if href and href.startswith('http'):
                                urls.append(href)
                                title_text = link.get_text(strip=True)
                                if title_text:
                                    titles.append(title_text)
                        
                        # Find snippets
                        for snippet in soup.find_all(class_=re.compile('result__snippet')):
                            snippet_text = snippet.get_text(strip=True)
                            if snippet_text:
                                snippets.append(snippet_text)
                    except ImportError:
                        pass
                
                # Combine results
                for i in range(min(len(urls), max_results)):
                    url = urls[i] if i < len(urls) else ""
                    title = titles[i] if i < len(titles) else ""
                    snippet = snippets[i] if i < len(snippets) else ""
                    
                    if url and url.startswith("http"):
                        results.append({
                            "url": url,
                            "title": title.strip() if title else "",
                            "snippet": snippet.strip() if snippet else ""
                        })
            except Exception as e:
                print(f"[RESEARCHER] HTML scraping fallback failed: {e}")
        
        return results[:max_results]
        
    except Exception as e:
        print(f"[RESEARCHER] DuckDuckGo search error: {e}")
        return []


def extract_with_jina_reader(url: str, max_length: int = 10000, use_local: bool = True) -> Optional[str]:
    """
    Extracts content from a URL using Jina Reader.
    
    Tries local Docker instance first (if enabled), then falls back to cloud API.
    
    Local Jina Reader API Format:
    - GET http://localhost:2222/{URL}
    - Response: Markdown text with metadata (Title, URL Source, Markdown Content)
    
    Args:
        url: The URL to extract content from
        max_length: Maximum content length
        use_local: If True, try local Jina Reader first (default: True)
    
    Returns:
        markdown content or None if extraction fails
    """
    from ..config import settings
    
    # Try local Jina Reader first (if enabled)
    if use_local and settings.JINA_READER_ENABLED:
        try:
            # Local Jina Reader: URL goes directly in the path
            # Format: GET http://localhost:2222/{URL}
            local_jina_url = f"{settings.JINA_READER_URL}/{url}"
            headers = {
                "Accept": "text/markdown"
            }
            
            response = requests.get(local_jina_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Local Jina returns markdown with metadata header:
            # Title: ...
            # URL Source: ...
            # Markdown Content:
            # ...
            content = response.text
            
            # Extract markdown content (skip metadata header if present)
            if "Markdown Content:" in content:
                # Find the start of actual markdown content
                markdown_start = content.find("Markdown Content:") + len("Markdown Content:")
                content = content[markdown_start:].strip()
            
            # Limit content length
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            print(f"[RESEARCHER] Extracted {len(content)} chars from {url} (local Jina)")
            return content
            
        except Exception as e:
            print(f"[RESEARCHER] Local Jina Reader failed for {url}: {e}")
            print("[RESEARCHER] Falling back to cloud Jina Reader API...")
    
    # Fallback: Cloud Jina Reader API
    try:
        cloud_jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "application/json",
            "X-Return-Format": "markdown"
        }
        
        response = requests.get(cloud_jina_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Jina returns markdown directly
        content = response.text
        
        # Limit content length
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        print(f"[RESEARCHER] Extracted {len(content)} chars from {url} (cloud Jina)")
        return content
        
    except Exception as e:
        print(f"[RESEARCHER] Jina Reader extraction error for {url}: {e}")
        return None


def extract_pdf_with_unstructured(url: str, max_length: int = 10000) -> Optional[str]:
    """
    Extracts content from a PDF URL using Unstructured.
    
    Supports both:
    - Direct library usage (if unstructured is installed)
    - API usage (if Unstructured API is running)
    
    Args:
        url: The PDF URL to extract content from
        max_length: Maximum content length
    
    Returns:
        Extracted text content or None if extraction fails
    """
    from ..config import settings
    import tempfile
    import os
    
    # Check if URL is a PDF
    if not url.lower().endswith('.pdf') and 'pdf' not in url.lower():
        return None
    
    if not settings.UNSTRUCTURED_ENABLED:
        return None
    
    try:
        # Download PDF to temporary file
        print(f"[RESEARCHER] Downloading PDF from {url}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        try:
            # Strategy 1: Try library first (if enabled or API not available)
            library_available = False
            if settings.UNSTRUCTURED_USE_LIBRARY:
                try:
                    from unstructured.partition.pdf import partition_pdf
                    library_available = True
                except (ImportError, ModuleNotFoundError) as e:
                    print(f"[RESEARCHER] Unstructured library not available (missing dependencies: {e}), trying API...")
                except Exception as e:
                    print(f"[RESEARCHER] Unstructured library import error: {e}, trying API...")
            
            if library_available:
                try:
                    print(f"[RESEARCHER] Extracting PDF with Unstructured library...")
                    elements = partition_pdf(
                        filename=tmp_path,
                        strategy="fast",  # Fast strategy for speed
                        infer_table_structure=False
                    )
                    
                    # Combine all text elements
                    text_parts = []
                    for element in elements:
                        if hasattr(element, 'text') and element.text:
                            text_parts.append(element.text)
                    
                    content = "\n\n".join(text_parts)
                    
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    
                    print(f"[RESEARCHER] Extracted {len(content)} chars from PDF (Unstructured library)")
                    return content
                    
                except Exception as lib_error:
                    print(f"[RESEARCHER] Unstructured library failed: {lib_error}")
                    if not settings.UNSTRUCTURED_USE_LIBRARY:
                        # If library was auto-tried, continue to API
                        pass
                    else:
                        # If library was explicitly enabled, don't try API
                        return None
            
            # Strategy 2: Fallback to API (if library not available or failed)
            if not settings.UNSTRUCTURED_USE_LIBRARY or not library_available:
                try:
                    from unstructured.partition.api import partition_via_api
                    
                    print(f"[RESEARCHER] Extracting PDF with Unstructured API...")
                    elements = partition_via_api(
                        filename=tmp_path,
                        api_url=settings.UNSTRUCTURED_API_URL,
                        api_key=settings.UNSTRUCTURED_API_KEY if settings.UNSTRUCTURED_API_KEY else None
                    )
                    
                    # Combine all text elements
                    text_parts = []
                    for element in elements:
                        if hasattr(element, 'text') and element.text:
                            text_parts.append(element.text)
                    
                    content = "\n\n".join(text_parts)
                    
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    
                    print(f"[RESEARCHER] Extracted {len(content)} chars from PDF (Unstructured API)")
                    return content
                    
                except ImportError:
                    print("[RESEARCHER] Unstructured API client not available")
                    return None
                except Exception as api_error:
                    print(f"[RESEARCHER] Unstructured API failed: {api_error}")
                    # If API failed and library wasn't tried, try library as last resort
                    if not library_available:
                        try:
                            from unstructured.partition.pdf import partition_pdf
                            print(f"[RESEARCHER] Trying Unstructured library as fallback...")
                            elements = partition_pdf(
                                filename=tmp_path,
                                strategy="fast",
                                infer_table_structure=False
                            )
                            text_parts = []
                            for element in elements:
                                if hasattr(element, 'text') and element.text:
                                    text_parts.append(element.text)
                            content = "\n\n".join(text_parts)
                            if len(content) > max_length:
                                content = content[:max_length] + "..."
                            print(f"[RESEARCHER] Extracted {len(content)} chars from PDF (Unstructured library fallback)")
                            return content
                        except Exception:
                            pass
                    return None
            
            # If we get here, both library and API failed
            print("[RESEARCHER] Both Unstructured library and API failed.")
            print("[RESEARCHER] To fix:")
            print("[RESEARCHER]   1. Install missing dependencies: pip install pdfminer.six")
            print("[RESEARCHER]   2. Or start Unstructured API: docker-compose up -d")
            print("[RESEARCHER]   3. Or set UNSTRUCTURED_USE_LIBRARY=false and configure API URL")
            return None
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
                
    except Exception as e:
        print(f"[RESEARCHER] PDF extraction error for {url}: {e}")
        return None


def extract_with_readability(url: str) -> Optional[str]:
    """
    Fallback: Simple content extraction using readability algorithm.
    Uses requests + basic HTML parsing.
    """
    try:
        import html2text
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Convert HTML to markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        content = h.handle(response.text)
        
        # Limit length
        if len(content) > 10000:
            content = content[:10000] + "..."
        
        return content
        
    except ImportError:
        # html2text not available - use basic text extraction
        try:
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) > 10000:
                text = text[:10000] + "..."
            
            return text
            
        except ImportError:
            print("[RESEARCHER] BeautifulSoup not available for fallback extraction")
            return None
    except Exception as e:
        print(f"[RESEARCHER] Readability extraction error for {url}: {e}")
        return None

