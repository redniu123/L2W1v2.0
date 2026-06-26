from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from l2w1.config import L2W1Paths, L2W1Secrets, Settings, load_settings

L2W1_ENV_KEYS = (
    "L2W1_DATA_ROOT",
    "L2W1_PAPER1_RUNS",
    "L2W1_MODEL_ROOT",
    "L2W1_RESULTS_ROOT",
    "L2W1_CONFIGS_ROOT",
    "L2W1_KEY_FILE",
)


def test_defaults_use_project_root_relative_paths(tmp_path: Path) -> None:
    settings = load_settings(project_root=tmp_path, env={})

    assert isinstance(settings, Settings)
    assert isinstance(settings.paths, L2W1Paths)
    assert isinstance(settings.secrets, L2W1Secrets)
    assert settings.paths.project_root == tmp_path
    assert settings.paths.data_root == tmp_path / "data" / "l2w1data"
    assert settings.paths.paper1_runs == tmp_path / "paper1_runs"
    assert settings.paths.model_root == tmp_path / "models"
    assert settings.paths.results_root == tmp_path / "results"
    assert settings.paths.configs_root == tmp_path / "configs"
    assert settings.secrets.provider_key_path == tmp_path / "key.txt"


def test_environment_variable_overrides_single_field(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    for key in L2W1_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("L2W1_DATA_ROOT", "/abs/x")

    settings = load_settings(project_root=tmp_path)

    assert settings.paths.data_root == Path("/abs/x")
    assert settings.paths.paper1_runs == tmp_path / "paper1_runs"


def test_dotenv_file_overrides_defaults_with_comments_blank_lines_and_quotes(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / ".env.test"
    dotenv_path.write_text(
        """
# comment

data_root="relative data"
paper1_runs=paper_runs_from_dotenv
ignored_key=ignored
results_root=results_from_dotenv # inline comment
""".lstrip(),
        encoding="utf-8",
    )

    settings = load_settings(project_root=tmp_path, env={}, dotenv_path=dotenv_path)

    assert settings.paths.data_root == tmp_path / "relative data"
    assert settings.paths.paper1_runs == tmp_path / "paper_runs_from_dotenv"
    assert settings.paths.results_root == tmp_path / "results_from_dotenv"
    assert settings.paths.model_root == tmp_path / "models"


def test_environment_variables_have_priority_over_dotenv(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env.test"
    dotenv_path.write_text("data_root=from_dotenv\n", encoding="utf-8")

    settings = load_settings(
        project_root=tmp_path,
        env={"L2W1_DATA_ROOT": "from_env"},
        dotenv_path=dotenv_path,
    )

    assert settings.paths.data_root == tmp_path / "from_env"


def test_relative_paths_are_resolved_against_project_root(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path,
        env={"L2W1_MODEL_ROOT": "relative/models"},
    )

    assert settings.paths.model_root == tmp_path / "relative" / "models"


def test_absolute_paths_are_preserved(tmp_path: Path) -> None:
    absolute_model_root = tmp_path / "absolute-models"

    settings = load_settings(
        project_root=tmp_path,
        env={"L2W1_MODEL_ROOT": str(absolute_model_root)},
    )

    assert settings.paths.model_root == absolute_model_root


def test_provider_key_path_defaults_to_project_key_without_opening_file(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    key_path = tmp_path / "key.txt"
    original_open = Path.open

    def fail_if_key_file_is_opened(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == key_path:
            raise AssertionError("key.txt must not be opened by load_settings")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_if_key_file_is_opened)

    settings = load_settings(project_root=tmp_path, env={})

    assert not key_path.exists()
    assert settings.secrets.provider_key_path == key_path


def test_yaml_layer_is_loaded_and_overridden_by_dotenv_and_env(tmp_path: Path) -> None:
    pytest.importorskip("yaml")

    config_path = tmp_path / "settings.yaml"
    dotenv_path = tmp_path / ".env.test"
    config_path.write_text(
        """
data_root: from_yaml
model_root: model_from_yaml
results_root: results_from_yaml
provider_key_path: yaml_key.txt
""".lstrip(),
        encoding="utf-8",
    )
    dotenv_path.write_text(
        """
data_root=from_dotenv
provider_key_path=dotenv_key.txt
""".lstrip(),
        encoding="utf-8",
    )

    settings = load_settings(
        project_root=tmp_path,
        env={"L2W1_DATA_ROOT": "from_env"},
        dotenv_path=dotenv_path,
        config_path=config_path,
    )

    assert settings.paths.data_root == tmp_path / "from_env"
    assert settings.paths.model_root == tmp_path / "model_from_yaml"
    assert settings.paths.results_root == tmp_path / "results_from_yaml"
    assert settings.secrets.provider_key_path == tmp_path / "dotenv_key.txt"
