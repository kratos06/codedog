import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_pull_request():
    """
    Creates and returns a mock Pull Request object for testing.
    
    This fixture creates a MagicMock instance configured with preset attributes:
        - pull_request_id: 123
        - repository_id: 456
        - pull_request_number: 42
        - title: "Test PR"
        - body: "PR description"
        - url: "https://github.com/test/repo/pull/42"
        - repository_name: "test/repo"
    
    The object's json method is set to return an empty JSON object string ("{}").
    """
    mock_pr = MagicMock()
    mock_pr.pull_request_id = 123
    mock_pr.repository_id = 456
    mock_pr.pull_request_number = 42
    mock_pr.title = "Test PR"
    mock_pr.body = "PR description"
    mock_pr.url = "https://github.com/test/repo/pull/42"
    mock_pr.repository_name = "test/repo"
    mock_pr.json.return_value = "{}"
    return mock_pr

@pytest.fixture
def mock_llm():
    """
    Create a mock language model for unit testing.
    
    Returns a MagicMock instance with a stubbed invoke method that always returns a
    dictionary containing a test response. This fixture simulates a language model
    for testing purposes.
    """
    mock = MagicMock()
    mock.invoke.return_value = {"text": "Test response"}
    return mock 