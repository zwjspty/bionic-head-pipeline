from uuid import uuid4

from bionic_head.core.history import ConversationHistoryStore


def test_history_store_appends_user_assistant_pair_in_order() -> None:
    session_id = uuid4()
    store = ConversationHistoryStore(max_turn_pairs=6, max_chars=3000)

    store.append_pair(session_id, user="我叫小张。", assistant="你好小张。")

    assert store.get(session_id) == [
        {"role": "user", "content": "我叫小张。"},
        {"role": "assistant", "content": "你好小张。"},
    ]
    assert store.metrics(session_id).turn_count == 1
    assert store.metrics(session_id).char_count == len("我叫小张。你好小张。")


def test_history_store_trims_oldest_pairs_by_max_turn_pairs() -> None:
    session_id = uuid4()
    store = ConversationHistoryStore(max_turn_pairs=1, max_chars=3000)

    store.append_pair(session_id, user="第一轮用户", assistant="第一轮回复")
    store.append_pair(session_id, user="第二轮用户", assistant="第二轮回复")

    assert store.get(session_id) == [
        {"role": "user", "content": "第二轮用户"},
        {"role": "assistant", "content": "第二轮回复"},
    ]
    assert store.metrics(session_id).turn_count == 1


def test_history_store_trims_oldest_pairs_by_max_chars_but_keeps_newest_pair() -> None:
    session_id = uuid4()
    store = ConversationHistoryStore(max_turn_pairs=6, max_chars=10)

    store.append_pair(session_id, user="旧用户很长", assistant="旧回复很长")
    store.append_pair(session_id, user="新用户", assistant="新回复")

    assert store.get(session_id) == [
        {"role": "user", "content": "新用户"},
        {"role": "assistant", "content": "新回复"},
    ]
    assert store.metrics(session_id).char_count <= 10


def test_history_store_disabled_is_noop() -> None:
    session_id = uuid4()
    store = ConversationHistoryStore(enabled=False, max_turn_pairs=6, max_chars=3000)

    store.append_pair(session_id, user="不会保存", assistant="也不会保存")

    assert store.get(session_id) == []
    assert store.metrics(session_id).turn_count == 0
    assert store.metrics(session_id).char_count == 0


def test_history_store_returns_copies_not_mutable_internal_state() -> None:
    session_id = uuid4()
    store = ConversationHistoryStore(max_turn_pairs=6, max_chars=3000)
    store.append_pair(session_id, user="用户", assistant="回复")

    snapshot = store.get(session_id)
    snapshot.append({"role": "user", "content": "外部污染"})

    assert store.get(session_id) == [
        {"role": "user", "content": "用户"},
        {"role": "assistant", "content": "回复"},
    ]
