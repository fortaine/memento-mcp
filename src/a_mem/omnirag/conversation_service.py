"""
Conversation Persistence Service for Memento V3

Manages chat history, session state, and user feedback.
Supports both local (file-based) and Cosmos DB backends.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class ToolCall:
    """A tool invocation within a message."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Message:
    """A single message in a conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool_calls: List[ToolCall] = field(default_factory=list)
    notes_referenced: List[str] = field(default_factory=list)  # Note IDs
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "notes_referenced": self.notes_referenced
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        tool_calls = [ToolCall(**tc) for tc in data.get("tool_calls", [])]
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            tool_calls=tool_calls,
            notes_referenced=data.get("notes_referenced", []),
            message_id=data.get("message_id", str(uuid.uuid4()))
        )


@dataclass
class Conversation:
    """A full conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    context: Optional[str] = None  # Session topic/context
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def message_count(self) -> int:
        return len(self.messages)
    
    def add_message(self, message: Message):
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def add_user_message(self, content: str) -> Message:
        """Convenience: Add a user message."""
        msg = Message(role="user", content=content)
        self.add_message(msg)
        return msg
    
    def add_assistant_message(
        self, 
        content: str, 
        tool_calls: Optional[List[ToolCall]] = None,
        notes_referenced: Optional[List[str]] = None
    ) -> Message:
        """Convenience: Add an assistant message."""
        msg = Message(
            role="assistant", 
            content=content,
            tool_calls=tool_calls or [],
            notes_referenced=notes_referenced or []
        )
        self.add_message(msg)
        return msg
    
    def to_dict(self) -> dict:
        return {
            "id": self.session_id,  # For Cosmos DB
            "session_id": self.session_id,
            "partitionKey": self.user_id or "memento",
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": {
                "message_count": self.message_count
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        session_id = data.get("session_id") or data.get("id") or str(uuid.uuid4())
        return cls(
            session_id=session_id,
            user_id=data.get("user_id"),
            messages=messages,
            context=data.get("context"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat())
        )
    
    def get_context_window(self, max_messages: int = 20) -> List[dict]:
        """Get recent messages for LLM context window."""
        recent = self.messages[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]


@dataclass
class Feedback:
    """User feedback on a response."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    rating: Literal["thumbs_up", "thumbs_down", "1", "2", "3", "4", "5"] = "3"
    feedback_text: Optional[str] = None
    query: Optional[str] = None
    strategy_used: Optional[Literal["db", "vector", "graph"]] = None
    notes_referenced: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "partitionKey": self.user_id or "memento",
            "session_id": self.session_id,
            "message_id": self.message_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "query": self.query,
            "strategy_used": self.strategy_used,
            "notes_referenced": self.notes_referenced,
            "timestamp": self.timestamp
        }


# ============================================
# Storage Backends
# ============================================

class ConversationStore(ABC):
    """Abstract base class for conversation storage."""
    
    @abstractmethod
    async def save_conversation(self, conversation: Conversation) -> str:
        """Save or update a conversation."""
        pass
    
    @abstractmethod
    async def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get a conversation by session ID."""
        pass
    
    @abstractmethod
    async def list_conversations(
        self, 
        user_id: Optional[str] = None, 
        limit: int = 20
    ) -> List[Conversation]:
        """List recent conversations."""
        pass
    
    @abstractmethod
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation."""
        pass
    
    @abstractmethod
    async def save_feedback(self, feedback: Feedback) -> str:
        """Save user feedback."""
        pass


class LocalConversationStore(ConversationStore):
    """
    File-based conversation storage for V2/local development.
    
    Stores conversations as JSON files in the data directory.
    """
    
    def __init__(self, data_dir: Path):
        self.conversations_dir = data_dir / "conversations"
        self.feedback_dir = data_dir / "feedback"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_conversation(self, conversation: Conversation) -> str:
        """Save conversation to JSON file."""
        file_path = self.conversations_dir / f"{conversation.session_id}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)
        
        return conversation.session_id
    
    async def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Load conversation from JSON file."""
        file_path = self.conversations_dir / f"{session_id}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return Conversation.from_dict(data)
    
    async def list_conversations(
        self, 
        user_id: Optional[str] = None, 
        limit: int = 20
    ) -> List[Conversation]:
        """List recent conversations from files."""
        conversations = []
        
        for file_path in sorted(
            self.conversations_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit * 2]:  # Get extra for filtering
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                conv = Conversation.from_dict(data)
                
                if user_id is None or conv.user_id == user_id:
                    conversations.append(conv)
                
                if len(conversations) >= limit:
                    break
            except Exception:
                continue  # Skip corrupted files
        
        return conversations
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete conversation file."""
        file_path = self.conversations_dir / f"{session_id}.json"
        
        if file_path.exists():
            file_path.unlink()
            return True
        
        return False
    
    async def save_feedback(self, feedback: Feedback) -> str:
        """Save feedback to JSON file."""
        file_path = self.feedback_dir / f"{feedback.id}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(feedback.to_dict(), f, indent=2, ensure_ascii=False)
        
        return feedback.id


# ============================================
# Conversation Service
# ============================================

class ConversationService:
    """
    High-level service for managing conversations.
    
    Provides session management, message history, and feedback collection.
    """
    
    def __init__(self, store: ConversationStore):
        self.store = store
        self._current_session: Optional[Conversation] = None
    
    async def start_session(
        self, 
        user_id: Optional[str] = None,
        context: Optional[str] = None
    ) -> Conversation:
        """Start a new conversation session."""
        self._current_session = Conversation(
            user_id=user_id,
            context=context
        )
        await self.store.save_conversation(self._current_session)
        return self._current_session
    
    async def resume_session(self, session_id: str) -> Optional[Conversation]:
        """Resume an existing conversation session."""
        self._current_session = await self.store.get_conversation(session_id)
        return self._current_session
    
    async def add_user_message(self, content: str) -> Message:
        """Add a user message to current session."""
        if not self._current_session:
            await self.start_session()
        
        # Type assertion - session exists after start_session
        assert self._current_session is not None
        msg = self._current_session.add_user_message(content)
        await self.store.save_conversation(self._current_session)
        return msg
    
    async def add_assistant_response(
        self,
        content: str,
        tool_calls: Optional[List[ToolCall]] = None,
        notes_referenced: Optional[List[str]] = None
    ) -> Message:
        """Add an assistant response to current session."""
        if not self._current_session:
            raise ValueError("No active session. Call start_session first.")
        
        msg = self._current_session.add_assistant_message(
            content=content,
            tool_calls=tool_calls,
            notes_referenced=notes_referenced
        )
        await self.store.save_conversation(self._current_session)
        return msg
    
    async def get_context(self, max_messages: int = 20) -> List[dict]:
        """Get conversation context for LLM."""
        if not self._current_session:
            return []
        return self._current_session.get_context_window(max_messages)
    
    async def submit_feedback(
        self,
        rating: Literal["thumbs_up", "thumbs_down", "1", "2", "3", "4", "5"],
        message_id: Optional[str] = None,
        feedback_text: Optional[str] = None,
        query: Optional[str] = None,
        strategy_used: Optional[Literal["db", "vector", "graph"]] = None,
        notes_referenced: Optional[List[str]] = None
    ) -> Feedback:
        """Submit user feedback on a response."""
        if not self._current_session:
            raise ValueError("No active session.")
        
        feedback = Feedback(
            session_id=self._current_session.session_id,
            message_id=message_id,
            user_id=self._current_session.user_id,
            rating=rating,
            feedback_text=feedback_text,
            query=query,
            strategy_used=strategy_used,
            notes_referenced=notes_referenced or []
        )
        
        await self.store.save_feedback(feedback)
        return feedback
    
    async def list_recent_sessions(
        self, 
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Conversation]:
        """List recent conversation sessions."""
        return await self.store.list_conversations(user_id, limit)
    
    @property
    def current_session(self) -> Optional[Conversation]:
        """Get the current active session."""
        return self._current_session
    
    @property
    def current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._current_session.session_id if self._current_session else None
