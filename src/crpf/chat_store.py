from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


@dataclass(frozen=True)
class StoredMessage:
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    metadata: dict[str, Any]


class ChatHistoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists conversations (
                    id text primary key,
                    title text not null,
                    created_at text not null,
                    updated_at text not null
                );

                create table if not exists messages (
                    id integer primary key autoincrement,
                    conversation_id text not null references conversations(id) on delete cascade,
                    role text not null,
                    content text not null,
                    created_at text not null,
                    metadata_json text not null default '{}'
                );

                create index if not exists idx_conversations_updated_at
                    on conversations(updated_at desc);
                create index if not exists idx_messages_conversation_id
                    on messages(conversation_id, id);
                """
            )

    def list_conversations(self, limit: int = 80) -> list[Conversation]:
        self.ensure_schema()
        with self.connect() as conn:
            rows = conn.execute(
                """
                select c.id, c.title, c.created_at, c.updated_at, count(m.id) as message_count
                from conversations c
                left join messages m on m.conversation_id = c.id
                group by c.id
                order by c.updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [conversation_from_row(row) for row in rows]

    def create_conversation(self, title: str) -> Conversation:
        self.ensure_schema()
        now = utc_now()
        conversation_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                """
                insert into conversations (id, title, created_at, updated_at)
                values (?, ?, ?, ?)
                """,
                (conversation_id, normalize_title(title), now, now),
            )
        return Conversation(
            id=conversation_id,
            title=normalize_title(title),
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        self.ensure_schema()
        with self.connect() as conn:
            row = conn.execute(
                """
                select c.id, c.title, c.created_at, c.updated_at, count(m.id) as message_count
                from conversations c
                left join messages m on m.conversation_id = c.id
                where c.id = ?
                group by c.id
                """,
                (conversation_id,),
            ).fetchone()
        return conversation_from_row(row) if row else None

    def delete_conversation(self, conversation_id: str) -> bool:
        self.ensure_schema()
        with self.connect() as conn:
            cursor = conn.execute("delete from conversations where id = ?", (conversation_id,))
        return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> StoredMessage:
        self.ensure_schema()
        if role not in {"user", "assistant"}:
            raise ValueError("message role must be user or assistant")
        now = utc_now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self.connect() as conn:
            cursor = conn.execute(
                """
                insert into messages (conversation_id, role, content, created_at, metadata_json)
                values (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, now, metadata_json),
            )
            conn.execute(
                "update conversations set updated_at = ? where id = ?",
                (now, conversation_id),
            )
            message_id = int(cursor.lastrowid)
        return StoredMessage(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            metadata=metadata or {},
        )

    def get_messages(self, conversation_id: str) -> list[StoredMessage]:
        self.ensure_schema()
        with self.connect() as conn:
            rows = conn.execute(
                """
                select id, conversation_id, role, content, created_at, metadata_json
                from messages
                where conversation_id = ?
                order by id
                """,
                (conversation_id,),
            ).fetchall()
        return [message_from_row(row) for row in rows]

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        return conn


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_title(text: str) -> str:
    title = " ".join(text.strip().split())
    if not title:
        return "新聊天"
    return title[:32]


def conversation_from_row(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=str(row["id"]),
        title=str(row["title"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        message_count=int(row["message_count"]),
    )


def message_from_row(row: sqlite3.Row) -> StoredMessage:
    try:
        metadata = json.loads(str(row["metadata_json"] or "{}"))
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return StoredMessage(
        id=int(row["id"]),
        conversation_id=str(row["conversation_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
        metadata=metadata,
    )
