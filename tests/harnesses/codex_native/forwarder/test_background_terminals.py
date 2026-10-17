"""Background-terminal count tests for the Codex forwarder."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.harnesses.codex_native import forwarder as fwd
from omnigent.harnesses.codex_native.bridge import (
    CodexNativeBridgeState,
    write_bridge_state,
)
from tests.harnesses.codex_native.forwarder._support import _RecordingClient


def _command(item_id: str, command: str, *, status: str = "inProgress") -> dict:
    return {
        "type": "commandExecution",
        "id": item_id,
        "status": status,
        "processId": "4242",
        "source": "unifiedExecStartup",
        "command": command,
        "cwd": "/tmp",
    }


def _item_event(method: str, item: dict) -> dict:
    return {
        "method": method,
        "params": {"threadId": "thread_123", "turnId": "turn_123", "item": item},
    }


def _statuses(client: _RecordingClient) -> list[dict]:
    return [body["data"] for _, body in client.posts if body["type"] == "external_session_status"]


def _count_updates(client: _RecordingClient) -> list[dict]:
    return [body["data"] for _, body in client.posts if body["type"] == "subagent.status"]


async def _drive(tmp_path: Path, client: _RecordingClient, events: list[dict]) -> None:
    usage = fwd._SessionUsageCoalescer(client, "conv_x")  # type: ignore[arg-type]
    tracker = fwd._CodexElicitationTaskTracker()
    state = fwd._CodexForwarderState(parent_session_id="conv_x")
    for event in events:
        await fwd._handle_event(
            client,  # type: ignore[arg-type]
            session_id="conv_x",
            bridge_dir=tmp_path,
            event=event,
            usage_coalescer=usage,
            elicitation_tracker=tracker,
            expected_thread_id="thread_123",
            forwarder_state=state,
        )


def _bridge_state(tmp_path: Path) -> None:
    write_bridge_state(
        tmp_path,
        CodexNativeBridgeState(
            session_id="conv_x",
            socket_path=str(tmp_path / "app-server.sock"),
            thread_id="thread_123",
            codex_home=str(tmp_path / "codex-home"),
        ),
    )


_TURN_STARTED = {
    "method": "turn/started",
    "params": {"threadId": "thread_123", "turn": {"id": "turn_123", "status": "inProgress"}},
}
_TURN_COMPLETED = {
    "method": "turn/completed",
    "params": {
        "threadId": "thread_123",
        "turn": {"id": "turn_123", "status": "completed", "items": []},
    },
}


@pytest.mark.asyncio
async def test_terminal_outliving_its_turn_is_counted_until_it_exits(tmp_path: Path) -> None:
    """
    The TUI's "1 background terminal running" reaches Omnigent and clears on exit.

    The exit lands between turns, where a parent may already have dispatched the
    next message, so it updates the count without posting another turn-ending idle.
    """
    _bridge_state(tmp_path)
    client = _RecordingClient()
    background = _command("exec-bg", "/bin/bash -lc 'sleep 30'")
    foreground = _command("exec-fg", "/bin/bash -lc 'sleep 1'")

    await _drive(
        tmp_path,
        client,
        [
            _TURN_STARTED,
            _item_event("item/started", background),
            _item_event("item/started", foreground),
            _item_event("item/completed", {**foreground, "status": "completed"}),
            _TURN_COMPLETED,
            _item_event("item/completed", {**background, "status": "completed"}),
        ],
    )

    assert _statuses(client) == [
        {"status": "running", "response_id": "codex_turn_123"},
        {
            "status": "idle",
            "background_task_count": 1,
            "background_tasks": [
                {
                    "id": "exec-bg",
                    "type": "shell",
                    "status": "running",
                    "command": "/bin/bash -lc 'sleep 30'",
                }
            ],
            "response_id": "codex_turn_123",
        },
    ]
    assert _count_updates(client) == [{"idle": True, "background_task_count": 0}]


@pytest.mark.asyncio
async def test_turn_without_lingering_terminals_posts_no_count(tmp_path: Path) -> None:
    """Terminals that exit within their turn leave status edges unchanged."""
    _bridge_state(tmp_path)
    client = _RecordingClient()
    foreground = _command("exec-fg", "/bin/bash -lc 'echo hi'")

    await _drive(
        tmp_path,
        client,
        [
            _TURN_STARTED,
            _item_event("item/started", foreground),
            _item_event("item/completed", {**foreground, "status": "completed"}),
            _TURN_COMPLETED,
        ],
    )

    assert _statuses(client) == [
        {"status": "running", "response_id": "codex_turn_123"},
        {"status": "idle", "response_id": "codex_turn_123"},
    ]


@pytest.mark.asyncio
async def test_terminal_exiting_during_a_later_turn_does_not_idle_the_session(
    tmp_path: Path,
) -> None:
    """A mid-turn exit waits for that turn's idle edge instead of posting its own."""
    _bridge_state(tmp_path)
    client = _RecordingClient()
    background = _command("exec-bg", "/bin/bash -lc 'sleep 30'")
    second_started = {
        "method": "turn/started",
        "params": {"threadId": "thread_123", "turn": {"id": "turn_456", "status": "inProgress"}},
    }
    second_completed = {
        "method": "turn/completed",
        "params": {
            "threadId": "thread_123",
            "turn": {"id": "turn_456", "status": "completed", "items": []},
        },
    }

    await _drive(
        tmp_path,
        client,
        [
            _TURN_STARTED,
            _item_event("item/started", background),
            _TURN_COMPLETED,
            second_started,
            _item_event("item/completed", {**background, "status": "completed"}),
            second_completed,
        ],
    )

    statuses = _statuses(client)
    assert [s["status"] for s in statuses] == ["running", "idle", "running", "idle"]
    assert statuses[1]["background_task_count"] == 1
    assert statuses[3]["background_task_count"] == 0
    assert _count_updates(client) == []


@pytest.mark.asyncio
async def test_rotation_clears_the_old_chats_terminal_count() -> None:
    """After ``/new`` the old chat stops showing terminals the TUI no longer lists."""
    client = _RecordingClient()
    state = fwd._CodexForwarderState(parent_session_id="conv_old")
    state.note_terminal_started(_command("exec-bg", "sleep 30"))
    assert state.background_tasks_for_edge() is not None

    await fwd._clear_rotated_terminals(client, "conv_old", state)  # type: ignore[arg-type]

    assert client.posts == [
        (
            "/v1/sessions/conv_old/events",
            {
                "type": "subagent.status",
                "data": {"idle": True, "background_task_count": 0},
            },
        )
    ]
    assert state.background_tasks_for_edge() is None


def test_running_terminal_command_is_bounded_for_display() -> None:
    """A huge heredoc command is clamped in the detail but still counted."""
    state = fwd._CodexForwarderState(parent_session_id="conv_x")
    state.note_terminal_started(_command("exec-bg", "x" * 5000))

    tasks = state.background_tasks_for_edge()

    assert tasks is not None and len(tasks) == 1
    assert tasks[0]["command"] == "x" * 512
