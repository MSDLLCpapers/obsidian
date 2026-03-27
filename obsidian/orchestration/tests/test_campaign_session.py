"""Tests for CampaignSession wrapper."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from obsidian.campaign import Campaign
from obsidian.parameters import ParamSpace, Param_Continuous, Target
from obsidian.orchestration import CampaignSession, SessionStatus


@pytest.fixture
def simple_campaign():
    """Create a simple campaign for testing."""
    params = [Param_Continuous("Temperature", -10, 30), Param_Continuous("Concentration", 10, 150)]
    X_space = ParamSpace(params)
    target = Target("Yield", aim="max")
    campaign = Campaign(X_space=X_space, target=target, seed=42)
    return campaign


@pytest.fixture
def campaign_session(simple_campaign):
    """Create a campaign session for testing."""
    return CampaignSession(campaign=simple_campaign, name="Test Session")


def test_campaign_session_initialization(campaign_session):
    """Test session initialization."""
    assert campaign_session.session_id is not None
    assert campaign_session.name == "Test Session"
    assert campaign_session.status == SessionStatus.CONFIGURED
    assert len(campaign_session.history) == 0


def test_initialize_workflow(campaign_session):
    """Test initialize operation."""
    X0 = campaign_session.initialize(m_initial=5, method="LHS")

    assert len(X0) == 5
    assert campaign_session.status == SessionStatus.INITIALIZED
    assert len(campaign_session.history) == 1
    assert campaign_session.history[0]["operation"] == "initialize"


def test_add_data_workflow(campaign_session):
    """Test add_data operation."""
    # Initialize first
    X0 = campaign_session.initialize(m_initial=5, method="LHS")

    # Add some fake data
    data = X0.copy()
    data["Yield"] = [80.0, 85.0, 90.0, 75.0, 88.0]

    rows_added = campaign_session.add_data(data)

    assert rows_added == 5
    assert campaign_session.campaign.m_exp == 5
    assert len(campaign_session.history) == 2  # initialize + add_data


def test_fit_workflow(campaign_session):
    """Test fit operation."""
    # Initialize and add data
    X0 = campaign_session.initialize(m_initial=5, method="LHS")
    data = X0.copy()
    data["Yield"] = [80.0, 85.0, 90.0, 75.0, 88.0]
    campaign_session.add_data(data)

    # Fit
    campaign_session.fit()

    assert campaign_session.status == SessionStatus.FITTED
    assert campaign_session.campaign.optimizer.is_fit
    assert "fit" in [h["operation"] for h in campaign_session.history]


def test_suggest_workflow(campaign_session):
    """Test suggest operation."""
    # Setup: initialize, add data, fit
    X0 = campaign_session.initialize(m_initial=5, method="LHS")
    data = X0.copy()
    data["Yield"] = [80.0, 85.0, 90.0, 75.0, 88.0]
    campaign_session.add_data(data)
    campaign_session.fit()

    # Suggest
    X_suggest, eval_suggest = campaign_session.suggest(m_batch=2, acquisition=["NEI"])

    assert len(X_suggest) == 2
    assert campaign_session.status == SessionStatus.FITTED  # Returns to fitted
    assert "suggest" in [h["operation"] for h in campaign_session.history]


def test_get_best(campaign_session):
    """Test get_best operation."""
    # Before any data
    best = campaign_session.get_best()
    assert best["X_best"] is None
    assert best["message"] == "No data yet"

    # After adding data
    X0 = campaign_session.initialize(m_initial=5, method="LHS")
    data = X0.copy()
    yields = [80.0, 85.0, 90.0, 75.0, 88.0]

    # Get the actual target name from the campaign
    target_name = campaign_session.campaign.y_names[0]
    data[target_name] = yields
    campaign_session.add_data(data)

    best = campaign_session.get_best()
    assert best["X_best"] is not None
    assert best["n_experiments"] == 5

    # Check response_max has correct key and value
    assert isinstance(best["response_max"], dict)
    assert target_name in best["response_max"]
    assert best["response_max"][target_name] == max(yields)  # Should be 90.0


def test_save_and_load_state(campaign_session):
    """Test state persistence."""
    # Setup session with data
    X0 = campaign_session.initialize(m_initial=3, method="LHS")
    data = X0.copy()
    data["Yield"] = [80.0, 85.0, 90.0]
    campaign_session.add_data(data)
    campaign_session.fit()

    # Save state
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "test_session"
        state = campaign_session.save_state(directory=save_dir)

        # Check files created
        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "campaign_state.json").exists()
        assert (save_dir / "history.jsonl").exists()
        assert (save_dir / "data.csv").exists()

        # Load state
        loaded_session = CampaignSession.load_state(state)

        assert loaded_session.session_id == campaign_session.session_id
        assert loaded_session.name == campaign_session.name
        assert loaded_session.status == campaign_session.status
        assert loaded_session.campaign.m_exp == 3
        assert len(loaded_session.history) == len(campaign_session.history)


def test_load_from_directory(campaign_session):
    """Test loading from directory structure."""
    # Setup and save
    X0 = campaign_session.initialize(m_initial=3, method="LHS")
    data = X0.copy()
    data["Yield"] = [80.0, 85.0, 90.0]
    campaign_session.add_data(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "test_session"
        campaign_session.save_state(directory=save_dir)

        # Load from directory
        loaded = CampaignSession.load_from_directory(save_dir)

        assert loaded.session_id == campaign_session.session_id
        assert loaded.campaign.m_exp == 3


def test_to_dict(campaign_session):
    """Test metadata export."""
    metadata = campaign_session.to_dict()

    assert "session_id" in metadata
    assert "name" in metadata
    assert "status" in metadata
    assert "n_experiments" in metadata
    assert metadata["n_parameters"] == 2
    assert metadata["n_targets"] == 1


def test_error_handling(campaign_session):
    """Test error handling in operations."""
    # Try to fit without data
    with pytest.raises(ValueError):
        campaign_session.fit()

    # Status should be ERROR
    assert campaign_session.status == SessionStatus.ERROR
    assert "fit_error" in [h["operation"] for h in campaign_session.history]
