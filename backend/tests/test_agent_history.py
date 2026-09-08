from unittest.mock import MagicMock

from app.agent.history import ensure_thread, save_message, list_threads, get_thread, _make_title


def test_make_title_truncates_long_first_message():
    long_text = "a" * 100
    title = _make_title(long_text)
    assert title.endswith("…")
    assert len(title) == 61  # 60 chars + ellipsis


def test_make_title_collapses_whitespace():
    assert _make_title("hello\n  world  ") == "hello world"


def test_make_title_empty_message_falls_back():
    assert _make_title("   ") == "New conversation"


def test_ensure_thread_creates_when_absent():
    db = MagicMock()
    db.get.return_value = None

    ensure_thread(db, "t1", "What is the ISS National Lab?")

    db.add.assert_called_once()
    added = db.add.call_args[0][0]
    assert added.id == "t1"
    assert added.title == "What is the ISS National Lab?"
    db.commit.assert_called_once()


def test_ensure_thread_noop_when_already_exists():
    db = MagicMock()
    db.get.return_value = MagicMock()  # thread already exists

    ensure_thread(db, "t1", "second message")

    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_save_message_adds_row_and_bumps_thread_updated_at():
    db = MagicMock()
    thread = MagicMock()
    db.get.return_value = thread

    save_message(db, "t1", "user", "hello", sources=[{"episode_id": "e1"}])

    db.add.assert_called_once()
    added = db.add.call_args[0][0]
    assert added.thread_id == "t1"
    assert added.role == "user"
    assert added.text == "hello"
    assert added.sources == [{"episode_id": "e1"}]
    assert thread.updated_at is not None
    db.commit.assert_called_once()


def test_save_message_survives_missing_thread():
    """No thread row to bump updated_at on — must not raise."""
    db = MagicMock()
    db.get.return_value = None

    save_message(db, "ghost-thread", "assistant", "reply")

    db.commit.assert_called_once()


def test_list_threads_orders_by_updated_at_desc():
    db = MagicMock()
    list_threads(db)

    db.query.assert_called_once()
    db.query.return_value.order_by.assert_called_once()


def test_get_thread_returns_none_when_absent():
    db = MagicMock()
    db.get.return_value = None
    assert get_thread(db, "nope") is None
