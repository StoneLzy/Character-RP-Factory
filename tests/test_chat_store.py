import tempfile
import unittest
from pathlib import Path

from crpf.chat_store import ChatHistoryStore, normalize_title


class ChatHistoryStoreTests(unittest.TestCase):
    def test_create_list_messages_and_delete_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChatHistoryStore(Path(tmpdir) / "chat_history.sqlite3")
            conversation = store.create_conversation("讲讲名古屋公演和咲季")

            store.add_message(conversation.id, "user", "讲讲名古屋公演")
            store.add_message(
                conversation.id,
                "assistant",
                "这是咲季的重要舞台。",
                {"kind": "chat-saki", "rag_used": True},
            )

            conversations = store.list_conversations()
            messages = store.get_messages(conversation.id)

            self.assertEqual(len(conversations), 1)
            self.assertEqual(conversations[0].message_count, 2)
            self.assertEqual(messages[0].role, "user")
            self.assertEqual(messages[1].metadata["rag_used"], True)
            self.assertTrue(store.delete_conversation(conversation.id))
            self.assertEqual(store.list_conversations(), [])

    def test_normalize_title_defaults_and_truncates(self):
        self.assertEqual(normalize_title(""), "新聊天")
        self.assertEqual(normalize_title("  今天   聊聊咲季  "), "今天 聊聊咲季")
        self.assertLessEqual(len(normalize_title("咲季" * 40)), 32)


if __name__ == "__main__":
    unittest.main()
