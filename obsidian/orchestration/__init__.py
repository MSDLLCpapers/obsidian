"""
Orchestration layer for managing optimization campaigns.

This module provides framework-agnostic classes for orchestrating Bayesian optimization
workflows. It can be used by REST APIs, Dash applications, CLI tools, or any other Python
application that needs to manage optimization campaigns.

Key Components:
    - SessionManager: Singleton class for managing multiple campaign sessions
    - CampaignSession: Wrapper around Campaign with lifecycle management
    - SessionStatus: Enum for session lifecycle states
"""

from obsidian.orchestration.enums import SessionStatus
from obsidian.orchestration.campaign_session import CampaignSession
from obsidian.orchestration.session_manager import SessionManager

__all__ = [
    "SessionStatus",
    "CampaignSession",
    "SessionManager",
]
