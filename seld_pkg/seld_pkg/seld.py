import pickle
import sys
from pathlib import Path
from typing import Optional

import extract_features as baseline_extract_features
import model as baseline_model
import rclpy
import torch
import utils as baseline_utils
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node

from seld_pkg import BASELINE_ROOT


class SeldInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("seld")
        self._model_dir = Path(
            f"{BASELINE_ROOT}/checkpoints/SELDnet_audio_visual_multiACCDOA_20250331_173131"
        ).resolve()
        self.param = self._load_params(Path(f"{self._model_dir}/config.pkl").resolve())

        self.output_dir = Path(self.param["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._feature_extractor = baseline_extract_features.SELDFeatureExtractor(
            self.param
        )
        self._seld_model = self._load_model()
        self._service = self.create_service(
            SetParameters, "run_inference", self.run_inference
        )

        self.get_logger().info("Service ready on 'run_inference'.")

    def _load_params(self, param_path: Path) -> dict:
        if not param_path.exists():
            raise FileNotFoundError(f"Config file not found: {param_path}")
        with open(param_path, "rb") as handle:
            return pickle.load(handle)

    def _load_model(self) -> torch.nn.Module:
        self.get_logger().info(f"Using model directory: {self._model_dir}")

        seld_model = baseline_model.SELDModel(self.param).to(self.device)
        model_file = Path(f"{self._model_dir}/best_model.pth").resolve()
        if not model_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

        model_ckpt = torch.load(
            f"{model_file}",
            map_location=self.device,
            weights_only=False,
        )
        seld_model.load_state_dict(model_ckpt["seld_model"])
        return seld_model

    def _prepare_audio_feature(self, audio_path: Path) -> torch.Tensor:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = baseline_utils.load_audio(
            str(audio_path), self._feature_extractor.sampling_rate
        )
        if audio is None or sr is None:
            raise ValueError(f"No audio data decoded: {audio_path}")
        audio_feat_np = baseline_utils.extract_log_mel_spectrogram(
            audio,
            sr,
            self._feature_extractor.n_fft,
            self._feature_extractor.hop_length,
            self._feature_extractor.win_length,
            self._feature_extractor.nb_mels,
        )
        audio_feat = torch.tensor(audio_feat_np, dtype=torch.float32)

        if audio_feat.ndim != 3:
            raise ValueError(
                f"Audio feature tensor must be 3D, got shape {tuple(audio_feat.shape)}"
            )

        if audio_feat.shape[0] == 2:
            audio_feat = audio_feat.unsqueeze(0)

        return audio_feat.to(self.device)

    def _prepare_video_feature(self, video_path: Path) -> torch.Tensor:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = baseline_utils.load_video(str(video_path), self._feature_extractor.fps)
        if not frames:
            raise ValueError(f"No frames decoded: {video_path}")
        video_feat = baseline_utils.extract_resnet_features(
            frames,
            self._feature_extractor.preprocess,
            self._feature_extractor.backbone,
            self.device,
        )

        if video_feat.ndim == 3:
            video_feat = video_feat.unsqueeze(0)

        if video_feat.ndim != 4:
            raise ValueError(
                f"Video feature tensor must be 4D, got shape {tuple(video_feat.shape)}"
            )

        return video_feat.to(self.device)

    def run_inference(
        self, request: SetParameters.Request, response: SetParameters.Response
    ) -> SetParameters.Response:
        audio_path = Path(request.parameters[0].value.string_value).resolve()
        video_path = Path(request.parameters[1].value.string_value).resolve()
        result = SetParametersResult()
        try:
            audio_features = self._prepare_audio_feature(audio_path)
            video_features = self._prepare_video_feature(video_path)
            with torch.no_grad():
                logits = self._seld_model(audio_features, video_features)

            filelist = [str(audio_path)]
            baseline_utils.write_logits_to_dcase_format(
                logits,
                self.param,
                str(self.output_dir),
                filelist,
                split="single-file",
            )
            result.successful = True
            result.reason = f"Result saved to {self.output_dir}"
            self.get_logger().info(f"Result saved to {self.output_dir}")
        except Exception as exc:
            result.successful = False
            result.reason = f"ERROR: {exc}"
            self.get_logger().error(f"ERROR: {exc}")
        response.results = [result]
        return response


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = SeldInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
