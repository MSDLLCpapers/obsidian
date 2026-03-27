"""
FastAPI HTTP adapter for Obsidian optimization campaigns.

This module provides a REST API built on FastAPI that exposes the orchestration
layer (SessionManager, CampaignSession) via HTTP endpoints.

Design: Thin adapter layer that converts HTTP requests to orchestration calls.
Business logic resides in the orchestration layer.
"""

__version__ = "0.1.0"
