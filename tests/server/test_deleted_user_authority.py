"""Deleting a user in accounts mode must revoke the durable authority that
lets that identity keep acting unattended: scheduled tasks, device/refresh
grants, hosts and session tokens — and a later scheduled fire must not
re-create the deleted account via ``permission_store.ensure_user``.

Both tests drive the production-shaped accounts app (``create_app`` + real
stores) through the real HTTP journey and the real scheduler fire path.
"""

from __future__ import annotations

import asyncio
import secrets
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient


@dataclass
class AccountsFixture:
    client: TestClient
    account_store: Any
    permission_store: Any
    host_store: Any
    scheduled_task_store: Any
    agent_store: Any
    conversation_store: Any
    agent_cache: Any


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AccountsFixture]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMNIGENT_DATA_DIR", str(tmp_path / ".omnigent"))
    monkeypatch.setenv("OMNIGENT_AUTH_PROVIDER", "accounts")
    monkeypatch.setenv("OMNIGENT_ACCOUNTS_COOKIE_SECRET", secrets.token_hex(32))
    monkeypatch.setenv("OMNIGENT_ACCOUNTS_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("OMNIGENT_ACCOUNTS_INIT_ADMIN_PASSWORD", "admin-pw-12345")
    monkeypatch.setenv("OMNIGENT_ACCOUNTS_INIT_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("OMNIGENT_ADMIN_CREDENTIALS_PATH", str(tmp_path / "admin-creds"))
    monkeypatch.setenv("OMNIGENT_ACCOUNTS_AUTO_OPEN", "0")
    monkeypatch.delenv("OMNIGENT_OIDC_ISSUER", raising=False)

    db_url = f"sqlite:///{tmp_path}/test.db"
    from omnigent.db.utils import get_or_create_engine
    from omnigent.runtime import init as init_runtime
    from omnigent.runtime import telemetry
    from omnigent.runtime.agent_cache import AgentCache
    from omnigent.runtime.caps import RuntimeCaps
    from omnigent.server.accounts_store import SqlAlchemyAccountStore
    from omnigent.server.app import create_app
    from omnigent.server.auth import create_auth_provider
    from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
    from omnigent.stores.artifact_store.local import LocalArtifactStore
    from omnigent.stores.comment_store.sqlalchemy_store import SqlAlchemyCommentStore
    from omnigent.stores.conversation_store.sqlalchemy_store import (
        SqlAlchemyConversationStore,
    )
    from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
    from omnigent.stores.host_store import HostStore
    from omnigent.stores.permission_store.sqlalchemy_store import (
        SqlAlchemyPermissionStore,
    )
    from omnigent.stores.scheduled_task_store.sqlalchemy_store import (
        SqlAlchemyScheduledTaskStore,
    )

    get_or_create_engine(db_url)
    telemetry.init()
    permission_store = SqlAlchemyPermissionStore(db_url)
    agent_store = SqlAlchemyAgentStore(db_url)
    conversation_store = SqlAlchemyConversationStore(db_url)
    file_store = SqlAlchemyFileStore(db_url)
    comment_store = SqlAlchemyCommentStore(db_url)
    host_store = HostStore(db_url)
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    agent_cache = AgentCache(artifact_store=artifact_store, cache_dir=tmp_path / "cache")
    init_runtime(
        agent_cache=agent_cache,
        caps=RuntimeCaps(),
        agent_store=agent_store,
        file_store=file_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
        comment_store=comment_store,
    )
    auth_provider = create_auth_provider()
    account_store = SqlAlchemyAccountStore(db_url)
    scheduled_task_store = SqlAlchemyScheduledTaskStore(db_url)
    app = create_app(
        agent_store=agent_store,
        file_store=file_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
        agent_cache=agent_cache,
        comment_store=comment_store,
        permission_store=permission_store,
        host_store=host_store,
        auth_provider=auth_provider,
        account_store=account_store,
    )
    with TestClient(app) as client:
        yield AccountsFixture(
            client=client,
            account_store=account_store,
            permission_store=permission_store,
            host_store=host_store,
            scheduled_task_store=scheduled_task_store,
            agent_store=agent_store,
            conversation_store=conversation_store,
            agent_cache=agent_cache,
        )


@pytest.fixture
def accounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AccountsFixture]:
    yield from _build(tmp_path, monkeypatch)


def _login(
    client: TestClient, username: str, password: str, *, issue_refresh: bool = False
) -> httpx.Response:
    client.cookies.clear()
    body: dict[str, Any] = {"username": username, "password": password}
    if issue_refresh:
        body["issue_refresh"] = True
    r = client.post("/auth/login", json=body)
    assert r.status_code == 200, r.text
    return r


def _as(
    client: TestClient, jar: dict[str, str], method: str, url: str, **kwargs: Any
) -> httpx.Response:
    """Issue a request authenticated as the owner of ``jar`` (cookie jar)."""
    client.cookies.clear()
    client.cookies.update(jar)
    resp = client.request(method, url, **kwargs)
    client.cookies.clear()
    return resp


def _create_alice_with_authority(fx: AccountsFixture, admin_jar: dict[str, str]) -> dict[str, Any]:
    """Register alice, log her in with a refresh grant, register her host, and
    create her scheduled task — the durable authority a delete must revoke."""
    client = fx.client

    r = _as(client, admin_jar, "POST", "/auth/invite", json={"is_admin": False})
    assert r.status_code == 200, r.text
    invite = r.json()["token"]

    client.cookies.clear()
    r = client.post(
        "/auth/register",
        json={"invite": invite, "username": "alice", "password": "alice-pw-12345"},
    )
    assert r.status_code == 200, r.text

    r = _login(client, "alice", "alice-pw-12345", issue_refresh=True)
    alice_jar = dict(r.cookies)
    refresh_token = r.json().get("refresh_token")
    assert refresh_token, f"login with issue_refresh must return a refresh_token: {r.json()}"
    client.cookies.clear()

    host_id = "host_" + uuid.uuid4().hex
    fx.host_store.upsert_on_connect(host_id, name="alice-laptop", user_id="alice")

    agent_id = "ag_" + uuid.uuid4().hex
    fx.agent_store.create(agent_id, name="alice-nightly", bundle_location="alice-nightly/none")
    task_id = uuid.uuid4().hex
    fx.scheduled_task_store.create(
        task_id,
        name="hourly",
        prompt="run agent X on my laptop",
        rrule="FREQ=HOURLY",
        user_id="alice",
        agent_id=agent_id,
        timezone="UTC",
        host_id=host_id,
        workspace=str(Path.home()),
        state="active",
    )

    # Precondition: alice's cookie authorizes the API before deletion.
    r = _as(client, alice_jar, "GET", "/v1/hosts")
    assert r.status_code == 200, f"alice should be authorized before deletion: {r.status_code}"

    return {
        "alice_jar": alice_jar,
        "refresh_token": refresh_token,
        "host_id": host_id,
        "agent_id": agent_id,
        "task_id": task_id,
    }


def test_delete_user_revokes_durable_authority(accounts: AccountsFixture) -> None:
    """Deleting alice must revoke every durable path that lets her keep acting."""
    fx = accounts
    client = fx.client
    admin_jar = dict(_login(client, "admin", "admin-pw-12345").cookies)
    state = _create_alice_with_authority(fx, admin_jar)

    r = _as(client, admin_jar, "DELETE", "/auth/users/alice")
    assert r.status_code == 204, f"delete should succeed: {r.status_code} {r.text}"

    assert fx.account_store.get_user("alice") is None, "user row should be gone"

    task = fx.scheduled_task_store.get(state["task_id"])
    assert task is not None and task.state == "deleted", (
        f"deleted user's scheduled task must be disabled, got state={task and task.state!r}"
    )

    client.cookies.clear()
    r = client.post(
        "/oauth/token",
        data={"grant_type": "refresh_token", "refresh_token": state["refresh_token"]},
    )
    assert r.status_code == 400, f"refresh grant must be revoked, got {r.status_code}: {r.text}"
    assert r.json().get("error") == "invalid_grant", r.text

    assert fx.host_store.list_hosts("alice") == [], "deleted user's hosts must be removed"

    # The auth provider itself must reject the token, so EVERY authenticated
    # route — not just /auth/me with its own existence guard — rejects it.
    r = _as(client, state["alice_jar"], "GET", "/v1/hosts")
    assert r.status_code == 401, (
        f"deleted user's session cookie must be rejected, got {r.status_code}"
    )


def test_scheduled_fire_does_not_resurrect_deleted_owner(accounts: AccountsFixture) -> None:
    """A scheduled fire for a deleted owner must not re-create the account."""
    fx = accounts
    client = fx.client
    admin_jar = dict(_login(client, "admin", "admin-pw-12345").cookies)
    state = _create_alice_with_authority(fx, admin_jar)

    r = _as(client, admin_jar, "DELETE", "/auth/users/alice")
    assert r.status_code == 204, r.text
    assert fx.account_store.get_user("alice") is None

    import omnigent.server.scheduled.fire as fire_mod
    from omnigent.server.scheduled.fire import FireDeps, build_on_fire

    deps = FireDeps(
        scheduled_task_store=fx.scheduled_task_store,
        agent_store=fx.agent_store,
        conversation_store=fx.conversation_store,
        permission_store=fx.permission_store,
        host_store=fx.host_store,
        host_registry=None,
        agent_cache=fx.agent_cache,
    )
    dispatched: list[str] = []

    async def fake_launch(conv: Any, task: Any) -> None:
        dispatched.append(conv.id)

    async def drive_fire() -> None:
        on_fire = build_on_fire(deps, launch_dispatch=fake_launch)
        await on_fire(0, state["task_id"])
        pending = list(fire_mod._PENDING_FIRES)
        if pending:
            await asyncio.gather(*pending)

    asyncio.new_event_loop().run_until_complete(drive_fire())

    assert fx.account_store.get_user("alice") is None, (
        "scheduled fire must not resurrect the deleted owner via ensure_user"
    )
    assert dispatched == [], "no run should be launched for a deleted owner"
