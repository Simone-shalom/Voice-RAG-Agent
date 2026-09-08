import uuid

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    source_url = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    chunks = relationship("Chunk", back_populates="episode", lazy="select")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id = Column(UUID(as_uuid=True), ForeignKey("episodes.id"), nullable=False)
    start_ts = Column(Float, nullable=False)
    end_ts = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    episode = relationship("Episode", back_populates="chunks")


class ChatThread(Base):
    __tablename__ = "chat_threads"

    # id is the LangGraph thread_id (client-generated crypto.randomUUID()
    # string), not a server-assigned uuid — a String primary key avoids
    # coupling this table to any particular id format the frontend picks.
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())
    messages = relationship(
        "ChatMessage", back_populates="thread", lazy="select", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String, ForeignKey("chat_threads.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant"
    text = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    thread = relationship("ChatThread", back_populates="messages")
