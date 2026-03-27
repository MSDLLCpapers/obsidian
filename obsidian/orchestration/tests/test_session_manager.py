"""Tests for SessionManager."""

import pytest
import tempfile
from pathlib import Path

from obsidian.parameters import ParamSpace, Param_Continuous, Target
from obsidian.orchestration import SessionManager, SessionStatus


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "sessions"


@pytest.fixture
def manager(temp_storage):
    """Create a fresh SessionManager for each test."""
    SessionManager.reset_instance()
    return SessionManager(storage_dir=temp_storage)


@pytest.fixture
def simple_setup():
    """Simple parameter space and target."""
    params = [Param_Continuous("Temp", 0, 100), Param_Continuous("Conc", 0, 200)]
    X_space = ParamSpace(params)
    target = Target("Yield", aim="max")
    return X_space, target


def test_singleton_pattern():
    """Test SessionManager singleton."""
    SessionManager.reset_instance()

    manager1 = SessionManager.get_instance()
    manager2 = SessionManager.get_instance()

    assert manager1 is manager2


def test_create_session(manager, simple_setup):
    """Test session creation."""
    X_space, target = simple_setup

    session = manager.create_session(X_space=X_space, target=target, name="Test Session", seed=42)

    assert session.session_id in manager.sessions
    assert session.name == "Test Session"
    assert session.status == SessionStatus.CONFIGURED

    # Check persistence
    session_dir = manager.storage_dir / session.session_id
    assert session_dir.exists()
    assert (session_dir / "metadata.json").exists()


def test_get_session(manager, simple_setup):
    """Test session retrieval."""
    X_space, target = simple_setup

    # Create session
    session = manager.create_session(X_space=X_space, target=target)
    session_id = session.session_id

    # Clear cache to force disk load
    manager.sessions.clear()

    # Retrieve session
    retrieved = manager.get_session(session_id)

    assert retrieved.session_id == session_id
    assert retrieved.session_id in manager.sessions  # Now cached


def test_get_nonexistent_session(manager):
    """Test retrieving non-existent session."""
    with pytest.raises(KeyError):
        manager.get_session("nonexistent-id")


def test_list_sessions(manager, simple_setup):
    """Test listing sessions."""
    X_space, target = simple_setup

    # Create multiple sessions
    session1 = manager.create_session(X_space=X_space, target=target, name="Session 1")
    session2 = manager.create_session(X_space=X_space, target=target, name="Session 2")

    # List all
    sessions = manager.list_sessions()

    assert len(sessions) >= 2
    session_ids = [s["session_id"] for s in sessions]
    assert session1.session_id in session_ids
    assert session2.session_id in session_ids


def test_list_sessions_with_filter(manager, simple_setup):
    """Test listing sessions with status filter."""
    X_space, target = simple_setup

    # Create sessions
    session1 = manager.create_session(X_space=X_space, target=target)
    session2 = manager.create_session(X_space=X_space, target=target)

    # Initialize one of them
    session1.initialize(m_initial=5)
    manager.save_session(session1.session_id)

    # List by status
    initialized = manager.list_sessions(status=SessionStatus.INITIALIZED)
    configured = manager.list_sessions(status=SessionStatus.CONFIGURED)

    assert any(s["session_id"] == session1.session_id for s in initialized)
    assert any(s["session_id"] == session2.session_id for s in configured)


def test_delete_session(manager, simple_setup):
    """Test session deletion."""
    X_space, target = simple_setup

    session = manager.create_session(X_space=X_space, target=target)
    session_id = session.session_id
    session_dir = manager.storage_dir / session_id

    # Delete
    manager.delete_session(session_id)

    assert session_id not in manager.sessions
    assert not session_dir.exists()


def test_delete_nonexistent_session(manager):
    """Test deleting non-existent session."""
    with pytest.raises(KeyError):
        manager.delete_session("nonexistent-id")


def test_save_session(manager, simple_setup):
    """Test explicit session save."""
    X_space, target = simple_setup

    session = manager.create_session(X_space=X_space, target=target)

    # Modify session
    session.initialize(m_initial=3)

    # Save
    manager.save_session(session.session_id)

    # Reload and check
    manager.sessions.clear()
    reloaded = manager.get_session(session.session_id)
    assert reloaded.status == SessionStatus.INITIALIZED


def test_session_exists(manager, simple_setup):
    """Test session existence check."""
    X_space, target = simple_setup

    session = manager.create_session(X_space=X_space, target=target)
    session_id = session.session_id

    assert manager.session_exists(session_id)
    assert not manager.session_exists("nonexistent-id")

    # Check after clearing cache
    manager.sessions.clear()
    assert manager.session_exists(session_id)


def test_cleanup_old_sessions(manager, simple_setup):
    """Test cleanup of old sessions."""
    X_space, target = simple_setup

    # Create sessions
    session1 = manager.create_session(X_space=X_space, target=target, name="Old")
    session2 = manager.create_session(X_space=X_space, target=target, name="New")

    # Manually set old timestamp
    import json
    from datetime import datetime, timedelta

    session1_dir = manager.storage_dir / session1.session_id
    metadata_file = session1_dir / "metadata.json"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    old_date = datetime.utcnow() - timedelta(days=35)
    metadata["updated_at"] = old_date.isoformat()

    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    # Cleanup sessions older than 30 days
    manager.cleanup_old_sessions(days=30)

    # Old session should be gone
    assert not manager.session_exists(session1.session_id)
    # New session should remain
    assert manager.session_exists(session2.session_id)


def test_storage_directory_creation(temp_storage):
    """Test that storage directory is created if it doesn't exist."""
    storage_path = temp_storage / "new_dir"
    assert not storage_path.exists()

    manager = SessionManager(storage_dir=storage_path)

    assert storage_path.exists()
