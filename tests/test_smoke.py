"""Tests for probflow package import and basic smoke tests."""

import importlib
import subprocess
import sys

import pytest


class TestImport:
    """Test that probflow can be imported."""

    def test_import_probflow(self) -> None:
        """Test importing the probflow package."""
        import probflow

        assert hasattr(probflow, "__version__")

    def test_version_exists(self) -> None:
        """Test that __version__ is defined."""
        import probflow

        assert hasattr(probflow, "__version__")
        assert isinstance(probflow.__version__, str)
        assert len(probflow.__version__) > 0

    def test_reimport(self) -> None:
        """Test that probflow can be reimported."""
        import probflow

        importlib.reload(probflow)
        assert probflow.__version__


class TestCLISmoke:
    """CLI smoke tests for the probflow package."""

    @pytest.mark.skipif(
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "probflow"],
            capture_output=True,
        ).returncode != 0,
        reason="probflow not installed via pip (run 'pip install -e .')",
    )
    def test_pip_show(self) -> None:
        """Test that pip show probflow succeeds."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "probflow"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "probflow" in result.stdout.lower()

    def test_python_c_import(self) -> None:
        """Test importing probflow via python -c."""
        result = subprocess.run(
            [sys.executable, "-c", "import probflow; print(probflow.__version__)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_python_c_version(self) -> None:
        """Test that version string is valid semver-like."""
        result = subprocess.run(
            [sys.executable, "-c", "import probflow; print(probflow.__version__)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        version = result.stdout.strip()
        # Basic semver check: at least major.minor.patch
        parts = version.split(".")
        assert len(parts) >= 3, f"Version {version!r} is not semver-like"
