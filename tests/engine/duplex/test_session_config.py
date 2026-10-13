# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU regressions for live duplex session configuration updates."""

from copy import deepcopy

import pytest

from vllm_omni.engine.duplex.config import DuplexConfigError, DuplexSessionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "voice_patch",
    [
        pytest.param({"voice": "verse"}, id="flat-voice"),
        pytest.param({"audio": {"output": {"voice": "verse"}}}, id="nested-voice"),
        pytest.param({"voice": None}, id="clear-voice"),
    ],
)
def test_voice_update_after_audio_started_preserves_config(voice_patch):
    config = DuplexSessionConfig(model="m", voice="alloy", instructions="original")
    before = deepcopy(config.as_dict())

    with pytest.raises(DuplexConfigError) as excinfo:
        config.apply_realtime_update(
            {"instructions": "changed", "tools": [{"type": "function", "name": "f"}], **voice_patch},
            audio_started=True,
        )

    assert excinfo.value.code == "voice_update_after_audio_unsupported"
    assert config.as_dict() == before


@pytest.mark.parametrize(
    ("voice_patch", "expected_voice"),
    [
        pytest.param({"voice": "verse"}, "verse", id="flat-voice"),
        pytest.param({"audio": {"output": {"voice": "verse"}}}, "verse", id="nested-voice"),
        pytest.param({"voice": None}, None, id="clear-voice"),
    ],
)
def test_voice_update_before_audio_started_is_allowed(voice_patch, expected_voice):
    config = DuplexSessionConfig(model="m", voice="alloy")

    config.apply_realtime_update(voice_patch, audio_started=False)

    assert config.voice == expected_voice


@pytest.mark.parametrize("voice", ["alloy", None])
def test_update_without_voice_after_audio_started_is_allowed(voice):
    config = DuplexSessionConfig(model="m", voice=voice, instructions="original")

    config.apply_realtime_update({"instructions": "changed"}, audio_started=True)

    assert config.voice == voice
    assert config.instructions == "changed"


def test_clearing_unset_voice_after_audio_started_is_allowed():
    config = DuplexSessionConfig(model="m", voice=None)

    config.apply_realtime_update({"voice": None, "instructions": "changed"}, audio_started=True)

    assert config.voice is None
    assert config.instructions == "changed"
