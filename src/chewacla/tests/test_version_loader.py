import pytest

from chewacla import _get_version


@pytest.mark.parametrize(
    "version_module, expected_version",
    [
        ({"version": "1.0.0"}, "1.0.0"),  # Test with a valid version module
        (None, "2.0.0"),  # Test with importlib.metadata returning a version
    ],
)
def test_get_version_with_version_module(mocker, version_module, expected_version):
    if version_module is not None:
        mock_version_module = mocker.Mock()
        mock_version_module.version = version_module["version"]
        version = _get_version(version_module=mock_version_module)
    else:
        # Mock importlib.metadata.version to return the expected version
        mocker.patch("importlib.metadata.version", return_value=expected_version)
        version = _get_version()

    assert version == expected_version


@pytest.mark.parametrize(
    "side_effect, expected_version",
    [
        (Exception("Not found"), "0+unknown"),  # Test fallback scenario
    ],
)
def test_get_version_fallback(mocker, side_effect, expected_version):
    # Mock importlib.metadata.version to raise an exception
    mocker.patch("importlib.metadata.version", side_effect=side_effect)

    version = _get_version()
    assert version == expected_version
