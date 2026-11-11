from pathlib import Path

import pytest

import src


@pytest.mark.parametrize(
    "resolver",
    [
        src.get_whole_spectra_phase_dir,
        src.get_cross_region_phase_dir,
        src.get_classification_phase_dir,
        src.get_spectral_interpretability_dir,
        src.get_ml_classification_phase_dir,
    ],
)
def test_phase_resolvers_reject_unknown_names_without_creating_outputs(
    resolver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setitem(src.config["output"], "eda_root", str(output))

    with pytest.raises(ValueError, match="Unknown analysis phase"):
        resolver("misspelled_phase")

    assert not output.exists()


def test_multiblock_phase_preserves_existing_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(src.config["output"], "eda_root", str(tmp_path))

    output = src.get_whole_spectra_phase_dir("exploratory_multiblock")

    assert output.name == "exploratory_multiblock"
    assert output.is_dir()
