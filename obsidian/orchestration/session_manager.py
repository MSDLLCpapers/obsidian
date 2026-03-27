"""
Session Manager for orchestrating multiple campaign sessions.

This module provides a singleton SessionManager that manages the lifecycle of
multiple CampaignSession objects. It handles:
- Session creation and deletion
- File-based persistence
- In-memory caching
- Session discovery and listing

Design: Framework-agnostic, no web/HTTP dependencies. Can be used by REST APIs,
Dash apps, CLI tools, or any Python application.
"""

import json
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from obsidian.campaign import Campaign
from obsidian.parameters import ParamSpace, Target
from obsidian.orchestration.campaign_session import CampaignSession
from obsidian.orchestration.enums import SessionStatus


class SessionManager:
    """
    Singleton manager for multiple campaign sessions.

    Manages session lifecycle including:
    - Creating new sessions
    - Loading existing sessions from disk
    - Caching active sessions in memory
    - Persisting sessions to disk
    - Listing and searching sessions
    - Cleaning up old sessions

    Attributes:
        storage_dir: Base directory for session storage
        sessions: In-memory cache of active sessions
    """

    _instance: Optional["SessionManager"] = None

    def __init__(self, storage_dir: Path | None = None):
        """
        Initialize SessionManager.

        Args:
            storage_dir: Base directory for session storage.
                        Defaults to ~/.obsidian/sessions
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".obsidian" / "sessions"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of sessions
        self.sessions: dict[str, CampaignSession] = {}

        # Load existing sessions on startup (metadata only for performance)
        self._discover_sessions()

    @classmethod
    def get_instance(cls, storage_dir: Path | None = None) -> "SessionManager":
        """
        Get singleton instance of SessionManager.

        Args:
            storage_dir: Storage directory (only used on first call)

        Returns:
            SessionManager singleton
        """
        if cls._instance is None:
            cls._instance = cls(storage_dir=storage_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (mainly for testing)."""
        cls._instance = None

    def _discover_sessions(self):
        """Discover existing sessions from storage directory (metadata only)."""
        if not self.storage_dir.exists():
            return

        for session_dir in self.storage_dir.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            session_id = metadata.get("session_id")
                            if session_id:
                                # Don't load full campaign yet (lazy loading)
                                # Just store the metadata for listing
                                pass
                    except Exception:
                        # Skip corrupted sessions
                        pass

    def create_session(
        self,
        X_space: ParamSpace,
        target: Target | list[Target],
        name: str | None = None,
        seed: int | None = None,
        **campaign_kwargs,
    ) -> CampaignSession:
        """
        Create a new campaign session.

        Args:
            X_space: Parameter space
            target: Target(s) for optimization
            name: Optional session name
            seed: Random seed
            **campaign_kwargs: Additional Campaign constructor kwargs

        Returns:
            New CampaignSession
        """
        # Create Campaign
        campaign = Campaign(X_space=X_space, target=target, seed=seed, **campaign_kwargs)

        # Wrap in CampaignSession
        session = CampaignSession(campaign=campaign, name=name, status=SessionStatus.CONFIGURED)

        # Cache in memory
        self.sessions[session.session_id] = session

        # Persist to disk
        session_dir = self.storage_dir / session.session_id
        session.save_state(directory=session_dir)

        return session

    def get_session(self, session_id: str) -> CampaignSession:
        """
        Get session by ID (loads from disk if not in cache).

        Args:
            session_id: Session ID

        Returns:
            CampaignSession

        Raises:
            KeyError: If session not found
        """
        # Check cache first
        if session_id in self.sessions:
            return self.sessions[session_id]

        # Try loading from disk
        session_dir = self.storage_dir / session_id
        if not session_dir.exists():
            raise KeyError(f"Session {session_id} not found")

        try:
            session = CampaignSession.load_from_directory(session_dir)
            # Cache it
            self.sessions[session_id] = session
            return session
        except Exception as e:
            raise KeyError(f"Failed to load session {session_id}: {e}")

    def list_sessions(self, status: SessionStatus | None = None) -> list[dict]:
        """
        List all sessions (metadata only).

        Args:
            status: Optional filter by status

        Returns:
            List of session metadata dictionaries
        """
        sessions_list = []

        for session_dir in self.storage_dir.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)

                        # Filter by status if requested
                        if status is not None:
                            if metadata.get("status") != str(status):
                                continue

                        # Enrich with data count if available
                        campaign_state_file = session_dir / "campaign_state.json"
                        if campaign_state_file.exists():
                            with open(campaign_state_file) as f:
                                campaign_state = json.load(f)
                                data_dict = campaign_state.get("data", {})
                                if data_dict:
                                    # Data is stored as dict with columns
                                    n_experiments = len(list(data_dict.values())[0]) if data_dict else 0
                                    metadata["n_experiments"] = n_experiments

                        sessions_list.append(metadata)
                    except Exception:
                        # Skip corrupted sessions
                        continue

        # Sort by updated_at (most recent first)
        sessions_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions_list

    def delete_session(self, session_id: str):
        """
        Delete a session (from memory and disk).

        Args:
            session_id: Session ID to delete

        Raises:
            KeyError: If session not found
        """
        # Remove from cache
        self.sessions.pop(session_id, None)

        # Remove from disk
        session_dir = self.storage_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        else:
            raise KeyError(f"Session {session_id} not found")

    def save_session(self, session_id: str):
        """
        Save session to disk.

        Args:
            session_id: Session ID to save

        Raises:
            KeyError: If session not found in cache
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not in cache")

        session_dir = self.storage_dir / session_id
        session.save_state(directory=session_dir)

    def cleanup_old_sessions(self, days: int = 30):
        """
        Delete sessions older than specified days.

        Args:
            days: Age threshold in days
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        for session_dir in self.storage_dir.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)

                        updated_at = datetime.fromisoformat(metadata.get("updated_at", ""))
                        if updated_at < cutoff:
                            session_id = metadata.get("session_id")
                            if session_id:
                                self.delete_session(session_id)
                    except Exception:
                        # Skip corrupted sessions
                        continue

    def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists (in cache or on disk).

        Args:
            session_id: Session ID

        Returns:
            True if session exists
        """
        if session_id in self.sessions:
            return True

        session_dir = self.storage_dir / session_id
        return session_dir.exists() and (session_dir / "metadata.json").exists()

    def __repr__(self) -> str:
        return f"SessionManager(storage_dir={self.storage_dir}, cached_sessions={len(self.sessions)})"
