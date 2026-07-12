"""Smoke tests: the base config loads and has the sections the rest of the code relies on."""

from fincrime_agents.config import load_config


def test_base_config_loads_and_has_expected_sections():
    cfg = load_config()
    assert cfg["llm"]["provider"] in {"ollama", "mock"}
    assert isinstance(cfg["llm"]["model"], str)
    assert cfg["search"]["provider"] in {"tavily", "offline"}
    assert isinstance(cfg["graph"]["recursion_limit"], int)
    assert cfg["paths"]["alerts_fixture"].endswith(".json")
