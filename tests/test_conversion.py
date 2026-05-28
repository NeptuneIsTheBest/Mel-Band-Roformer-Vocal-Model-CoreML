from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import coremltools as ct
from ml_collections import ConfigDict

from melband_roformer_coreml.conversion import convert_to_coreml, write_waveform_metadata


class ConversionTest(unittest.TestCase):
    def test_convert_uses_sliced_sdpa_by_default(self) -> None:
        mlmodel = mock.Mock()
        with mock.patch("melband_roformer_coreml.conversion.ct.convert", return_value=mlmodel) as convert:
            convert_to_coreml(
                traced=mock.Mock(),
                inputs=[ct.TensorType(name="audio", shape=(1, 2, 352800))],
                output_names=["vocals"],
                output_path=Path("model.mlpackage"),
            )

        kwargs = convert.call_args.kwargs
        pass_pipeline = kwargs["pass_pipeline"]
        self.assertEqual(kwargs["compute_precision"], ct.precision.FLOAT16)
        self.assertIn("common::scaled_dot_product_attention_sliced_q", pass_pipeline.passes)
        options = pass_pipeline.get_options("common::scaled_dot_product_attention_sliced_q")
        self.assertIsNotNone(options)
        self.assertEqual(
            {option.option_name: option.option_val for option in options},
            {
                "min_seq_length": 128,
                "seq_length_divider": 32,
            },
        )
        mlmodel.save.assert_called_once_with("model.mlpackage")

    def test_write_waveform_metadata_records_conversion_settings(self) -> None:
        config = ConfigDict(
            {
                "model": {
                    "sample_rate": 44100,
                    "stft_n_fft": 2048,
                    "stft_hop_length": 441,
                    "stft_win_length": 2048,
                    "stft_normalized": False,
                },
                "inference": {
                    "chunk_size": 352800,
                    "num_overlap": 2,
                },
            }
        )
        model = SimpleNamespace(audio_channels=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metadata.json"
            write_waveform_metadata(
                path,
                config,
                model,
                compute_precision_name="FLOAT16",
                use_sliced_sdpa=True,
                sdpa_min_seq_length=128,
                sdpa_seq_length_divider=32,
            )
            metadata = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["coreml"]["compute_precision"], "FLOAT16")
        self.assertEqual(metadata["coreml"]["attention"], "sliced scaled_dot_product_attention over Q")
        self.assertEqual(
            metadata["coreml"]["sliced_sdpa"],
            {
                "enabled": True,
                "min_seq_length": 128,
                "seq_length_divider": 32,
            },
        )


if __name__ == "__main__":
    unittest.main()
