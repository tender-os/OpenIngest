# SPDX-License-Identifier: Apache-2.0
"""Tests for model_provider_utils.py — client construction and config helpers."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from tensorlake_docai.providers.model_provider_utils import (
    _get_api_semaphore,
    _load_azure_openai_config,
    get_openai_client_and_model,
    get_openai_sync_client_and_model,
)

# ---------------------------------------------------------------------------
# _load_azure_openai_config
# ---------------------------------------------------------------------------


def test_load_azure_config_disabled_by_default(monkeypatch):
    monkeypatch.delenv("USE_AZURE_OPENAI", raising=False)
    assert _load_azure_openai_config() is None


def test_load_azure_config_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("USE_AZURE_OPENAI", "false")
    assert _load_azure_openai_config() is None


def test_load_azure_config_enabled_returns_tuple(monkeypatch):
    monkeypatch.setenv("USE_AZURE_OPENAI", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.azure.openai.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    result = _load_azure_openai_config()
    assert result == ("https://my.azure.openai.com", "secret-key", "gpt-4o")


def test_load_azure_config_enabled_missing_vars_raises(monkeypatch):
    from tensorlake.applications import RequestError as RequestException

    monkeypatch.setenv("USE_AZURE_OPENAI", "true")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", raising=False)
    with pytest.raises(RequestException):
        _load_azure_openai_config()


def test_load_azure_config_enabled_partial_vars_raises(monkeypatch):
    from tensorlake.applications import RequestError as RequestException

    monkeypatch.setenv("USE_AZURE_OPENAI", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.azure.openai.com")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", raising=False)
    with pytest.raises(RequestException):
        _load_azure_openai_config()


# ---------------------------------------------------------------------------
# get_openai_client_and_model (async client)
# ---------------------------------------------------------------------------


def test_get_openai_client_regular(monkeypatch):
    monkeypatch.delenv("USE_AZURE_OPENAI", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_client = MagicMock()
    with patch("langfuse.openai.AsyncOpenAI", return_value=mock_client):
        client, model = get_openai_client_and_model(api_key="test-key")
        assert model is not None
        assert isinstance(model, str)


def test_get_openai_client_azure_path(monkeypatch):
    monkeypatch.setenv("USE_AZURE_OPENAI", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.azure.openai.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "gpt-4-deploy")

    mock_azure_client = MagicMock()
    with patch("langfuse.openai.AsyncAzureOpenAI", return_value=mock_azure_client):
        client, model = get_openai_client_and_model()
        assert model == "gpt-4-deploy"


def test_get_openai_sync_client_azure_path(monkeypatch):
    monkeypatch.setenv("USE_AZURE_OPENAI", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.azure.openai.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "gpt-4-deploy")

    mock_azure_client = MagicMock()
    with patch("langfuse.openai.AzureOpenAI", return_value=mock_azure_client):
        client, model = get_openai_sync_client_and_model()
        assert model == "gpt-4-deploy"


def test_get_openai_sync_client_regular(monkeypatch):
    monkeypatch.delenv("USE_AZURE_OPENAI", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_client = MagicMock()
    with patch("langfuse.openai.OpenAI", return_value=mock_client):
        client, model = get_openai_sync_client_and_model(api_key="test-key")
        assert isinstance(model, str)


# ---------------------------------------------------------------------------
# _get_api_semaphore
# ---------------------------------------------------------------------------


def test_get_api_semaphore_outside_loop_returns_none():
    # Called outside an async context should return None gracefully
    result = _get_api_semaphore()
    assert result is None


def test_get_api_semaphore_inside_loop_returns_semaphore():
    async def _inner():
        sem = _get_api_semaphore()
        assert isinstance(sem, asyncio.Semaphore)
        # Second call returns the same semaphore (cached on loop)
        sem2 = _get_api_semaphore()
        assert sem is sem2

    asyncio.run(_inner())


def test_get_api_semaphore_limit_is_10():
    async def _inner():
        sem = _get_api_semaphore()
        assert sem._value == 10

    asyncio.run(_inner())
