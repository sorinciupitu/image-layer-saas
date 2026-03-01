from __future__ import annotations

import bisect
import inspect
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen-Image-Layered")
ENABLE_HEURISTIC_FALLBACK = os.getenv("ENABLE_HEURISTIC_FALLBACK", "false").lower() in {
    "1",
    "true",
    "yes",
}
LOW_VRAM_THRESHOLD_GB = float(os.getenv("LOW_VRAM_THRESHOLD_GB", "20"))
DEFAULT_INFERENCE_STEPS = int(os.getenv("DEFAULT_INFERENCE_STEPS", "24"))
DEFAULT_RESOLUTION = int(os.getenv("DEFAULT_RESOLUTION", "512"))
DEFAULT_CFG_SCALE = float(os.getenv("DEFAULT_CFG_SCALE", "3.0"))


@dataclass(frozen=True)
class ModelRuntimeInfo:
    model_id: str
    device: str
    dtype: str
    loaded: bool


@dataclass(frozen=True)
class InferenceOptions:
    num_inference_steps: int
    resolution: int
    true_cfg_scale: float
    use_en_prompt: bool
    cfg_normalize: bool

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "InferenceOptions":
        data = raw or {}
        steps = _coerce_int(data.get("num_inference_steps"), DEFAULT_INFERENCE_STEPS)
        resolution = _coerce_int(data.get("resolution"), DEFAULT_RESOLUTION)
        cfg = _coerce_float(data.get("true_cfg_scale"), DEFAULT_CFG_SCALE)
        use_en_prompt = _coerce_bool(data.get("use_en_prompt"), True)
        cfg_normalize = _coerce_bool(data.get("cfg_normalize"), True)

        steps = max(8, min(80, steps))
        resolution = max(256, min(1024, resolution))
        resolution = int(round(resolution / 64) * 64)
        cfg = max(1.0, min(8.0, cfg))

        return cls(
            num_inference_steps=steps,
            resolution=resolution,
            true_cfg_scale=cfg,
            use_en_prompt=use_en_prompt,
            cfg_normalize=cfg_normalize,
        )


class LayerDecompositionModel:
    _instance: "LayerDecompositionModel | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._pipeline: Any | None = None
        self._load_lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._allow_fallback = ENABLE_HEURISTIC_FALLBACK
        self._cuda_vram_gb = self._detect_cuda_vram_gb()
        self._low_vram_mode = self._device == "cuda" and self._cuda_vram_gb < LOW_VRAM_THRESHOLD_GB

    @classmethod
    def instance(cls, model_id: str = DEFAULT_MODEL_ID) -> "LayerDecompositionModel":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(model_id=model_id)
            return cls._instance

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def runtime_info(self) -> ModelRuntimeInfo:
        return ModelRuntimeInfo(
            model_id=self.model_id,
            device=self._device,
            dtype=str(self._dtype).replace("torch.", ""),
            loaded=self.is_loaded(),
        )

    def ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            self._pipeline = self._load_pipeline()
            LOGGER.info(
                "Loaded decomposition model '%s' on %s with dtype=%s",
                self.model_id,
                self._device,
                self._dtype,
            )

    def decompose_image(
        self,
        image: Image.Image,
        num_layers: int,
        options: dict[str, Any] | None = None,
    ) -> list[Image.Image]:
        if not 2 <= num_layers <= 8:
            raise ValueError("num_layers must be between 2 and 8")

        rgba_image = image.convert("RGBA")
        inference_options = InferenceOptions.from_raw(options)
        started_at = time.perf_counter()

        try:
            self.ensure_loaded()
            layers = self._run_model_inference(
                image=rgba_image,
                num_layers=num_layers,
                options=inference_options,
            )
        except Exception as exc:
            if self._allow_fallback:
                LOGGER.exception("Model inference failed; using deterministic fallback: %s", exc)
                layers = self._fallback_decompose(rgba_image, num_layers)
            else:
                LOGGER.exception("Model inference failed (fallback disabled): %s", exc)
                raise RuntimeError(f"Qwen-Image-Layered inference failed: {exc}") from exc

        elapsed = time.perf_counter() - started_at
        LOGGER.info("Image decomposition completed in %.2fs (layers=%d)", elapsed, len(layers))
        return layers

    def _load_pipeline(self) -> Any:
        self._apply_torch_custom_op_compatibility_shim()
        try:
            from diffusers import QwenImageLayeredPipeline
        except Exception as exc:
            raise RuntimeError(
                "QwenImageLayeredPipeline is unavailable. Install latest diffusers from GitHub "
                "or a compatible release with Qwen-Image-Layered support. "
                "If using torch 2.4.x, enable the custom-op compatibility shim."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "torch_dtype": self._dtype,
            "low_cpu_mem_usage": True,
        }

        if self._device == "cuda":
            load_kwargs["device_map"] = "balanced" if self._low_vram_mode else "cuda"
            load_kwargs["variant"] = "fp16"
        else:
            load_kwargs["device_map"] = "cpu"

        try:
            pipeline = QwenImageLayeredPipeline.from_pretrained(self.model_id, **load_kwargs)
        except Exception as exc:
            if "variant" not in load_kwargs:
                raise
            LOGGER.warning(
                "Failed to load model with variant=%s (%s). Retrying without variant.",
                load_kwargs.get("variant"),
                exc,
            )
            load_kwargs.pop("variant", None)
            pipeline = QwenImageLayeredPipeline.from_pretrained(self.model_id, **load_kwargs)

        if self._device == "cuda" and not load_kwargs.get("device_map"):
            pipeline.to("cuda")
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    LOGGER.debug("xFormers optimization not available for this pipeline")

        if self._low_vram_mode and hasattr(pipeline, "enable_model_cpu_offload"):
            try:
                pipeline.enable_model_cpu_offload()
                LOGGER.warning(
                    "Enabled low-VRAM mode with CPU offload (GPU VRAM %.2f GB < %.2f GB threshold).",
                    self._cuda_vram_gb,
                    LOW_VRAM_THRESHOLD_GB,
                )
            except Exception as exc:
                LOGGER.warning("Failed to enable model CPU offload: %s", exc)

        return pipeline

    def _detect_cuda_vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            total_bytes = float(torch.cuda.get_device_properties(0).total_memory)
            return total_bytes / (1024**3)
        except Exception:
            return 0.0

    def _apply_torch_custom_op_compatibility_shim(self) -> None:
        """
        Diffusers `main` can register torch custom ops during import.
        On torch 2.4.x this may fail with `infer_schema(... unsupported type torch.Tensor)`.
        For this runtime we monkey-patch custom-op decorators to no-op wrappers.
        """
        version = getattr(torch, "__version__", "")
        if not version.startswith("2.4"):
            return

        if os.getenv("DIFFUSERS_DISABLE_CUSTOM_OP_SHIM", "false").lower() in {"1", "true", "yes"}:
            return

        if not hasattr(torch, "library"):
            return

        try:
            current_custom_op = getattr(torch.library, "custom_op")
            if getattr(current_custom_op, "__name__", "") == "_diffusers_noop_custom_op":
                return
        except Exception:
            return

        def _diffusers_noop_custom_op(
            name: str,
            fn: Any | None = None,
            /,
            *,
            mutates_args: Any = (),
            device_types: Any = None,
            schema: str | None = None,
        ) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator if fn is None else fn

        def _diffusers_noop_register_fake(
            op: Any,
            fn: Any | None = None,
            /,
            *,
            lib: Any = None,
            _stacklevel: int = 1,
        ) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator if fn is None else fn

        torch.library.custom_op = _diffusers_noop_custom_op  # type: ignore[assignment]
        torch.library.register_fake = _diffusers_noop_register_fake  # type: ignore[assignment]
        LOGGER.warning(
            "Applied torch 2.4 custom-op compatibility shim for diffusers import "
            "(set DIFFUSERS_DISABLE_CUSTOM_OP_SHIM=true to disable)."
        )

    def _run_model_inference(
        self,
        image: Image.Image,
        num_layers: int,
        options: InferenceOptions,
    ) -> list[Image.Image]:
        if self._pipeline is None:
            raise RuntimeError("Model pipeline is not loaded")

        kwargs = self._build_inference_kwargs(image=image, num_layers=num_layers, options=options)
        output: Any
        with torch.inference_mode():
            if self._device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self._pipeline(**kwargs)
            else:
                output = self._pipeline(**kwargs)

        extracted_layers = self._extract_layers(output)
        normalized_layers = self._normalize_layers(extracted_layers, image.size, num_layers)
        if not normalized_layers:
            raise RuntimeError("Model returned no layers")
        return normalized_layers

    def _build_inference_kwargs(
        self,
        image: Image.Image,
        num_layers: int,
        options: InferenceOptions,
    ) -> dict[str, Any]:
        if self._pipeline is None:
            raise RuntimeError("Model pipeline is not loaded")

        signature = inspect.signature(self._pipeline.__call__)
        params = signature.parameters
        kwargs: dict[str, Any] = {}

        if "image" in params:
            kwargs["image"] = image
        elif "input_image" in params:
            kwargs["input_image"] = image
        elif "init_image" in params:
            kwargs["init_image"] = image
        else:
            kwargs["image"] = image

        if "layers" in params:
            kwargs["layers"] = num_layers
        elif "num_layers" in params:
            kwargs["num_layers"] = num_layers
        elif "n_layers" in params:
            kwargs["n_layers"] = num_layers

        if "output_type" in params:
            kwargs["output_type"] = "pil"
        if "return_dict" in params:
            kwargs["return_dict"] = True
        if "negative_prompt" in params:
            kwargs["negative_prompt"] = " "
        if "num_images_per_prompt" in params:
            kwargs["num_images_per_prompt"] = 1
        if "num_inference_steps" in params:
            kwargs["num_inference_steps"] = options.num_inference_steps
        if "true_cfg_scale" in params:
            kwargs["true_cfg_scale"] = options.true_cfg_scale
        if "cfg_normalize" in params:
            kwargs["cfg_normalize"] = options.cfg_normalize
        if "use_en_prompt" in params:
            kwargs["use_en_prompt"] = options.use_en_prompt
        if "resolution" in params:
            kwargs["resolution"] = options.resolution

        return kwargs

    def _extract_layers(self, output: Any) -> list[Image.Image]:
        direct_layers = self._coerce_layer_list(getattr(output, "images", None))
        if direct_layers:
            return direct_layers

        candidates: list[Any] = []
        self._collect_candidates(output, candidates)

        layer_images: list[Image.Image] = []
        for item in candidates:
            if isinstance(item, Image.Image):
                layer_images.append(item.convert("RGBA"))
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    if isinstance(sub_item, Image.Image):
                        layer_images.append(sub_item.convert("RGBA"))
        direct_from_candidates = self._coerce_layer_list(candidates)
        if direct_from_candidates:
            return direct_from_candidates
        return layer_images

    def _coerce_layer_list(self, value: Any) -> list[Image.Image]:
        if value is None:
            return []
        if isinstance(value, Image.Image):
            return [value.convert("RGBA")]
        if not isinstance(value, (list, tuple)):
            return []
        if not value:
            return []

        if all(isinstance(item, Image.Image) for item in value):
            return [item.convert("RGBA") for item in value]

        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            nested = value[0]
            if all(isinstance(item, Image.Image) for item in nested):
                return [item.convert("RGBA") for item in nested]

        flattened: list[Image.Image] = []
        for item in value:
            if isinstance(item, Image.Image):
                flattened.append(item.convert("RGBA"))
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    if isinstance(sub_item, Image.Image):
                        flattened.append(sub_item.convert("RGBA"))
        return flattened

    def _collect_candidates(self, value: Any, sink: list[Any]) -> None:
        if value is None:
            return
        if isinstance(value, Image.Image):
            sink.append(value)
            return
        if isinstance(value, dict):
            for key in ("layers", "images", "output", "outputs", "result"):
                if key in value:
                    self._collect_candidates(value[key], sink)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._collect_candidates(item, sink)
            return
        for attr in ("layers", "images", "output", "outputs", "result"):
            if hasattr(value, attr):
                self._collect_candidates(getattr(value, attr), sink)

    def _normalize_layers(
        self,
        layers: Sequence[Image.Image],
        size: tuple[int, int],
        expected_layers: int,
    ) -> list[Image.Image]:
        normalized: list[Image.Image] = [self._resize_to_canvas(layer.convert("RGBA"), size) for layer in layers]
        if len(normalized) == 1:
            split_layers = self._split_if_tiled(normalized[0], size, expected_layers)
            if split_layers:
                normalized = split_layers

        if len(normalized) >= expected_layers:
            return normalized[:expected_layers]

        while len(normalized) < expected_layers:
            normalized.append(Image.new("RGBA", size, (0, 0, 0, 0)))
        return normalized

    def _resize_to_canvas(self, image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image
        return image.resize(size, Image.Resampling.LANCZOS)

    def _split_if_tiled(
        self,
        image: Image.Image,
        size: tuple[int, int],
        expected_layers: int,
    ) -> list[Image.Image]:
        width, height = image.size
        canvas_w, canvas_h = size

        if width == canvas_w * expected_layers and height == canvas_h:
            return [
                image.crop((canvas_w * idx, 0, canvas_w * (idx + 1), canvas_h)).convert("RGBA")
                for idx in range(expected_layers)
            ]

        if height == canvas_h * expected_layers and width == canvas_w:
            return [
                image.crop((0, canvas_h * idx, canvas_w, canvas_h * (idx + 1))).convert("RGBA")
                for idx in range(expected_layers)
            ]

        return []

    def _fallback_decompose(self, image: Image.Image, num_layers: int) -> list[Image.Image]:
        width, height = image.size
        alpha_channel = image.split()[-1]
        opaque_values = [px for px in alpha_channel.getdata() if px > 0]
        if not opaque_values:
            return [Image.new("RGBA", image.size, (0, 0, 0, 0)) for _ in range(num_layers)]

        thresholds = self._build_luma_thresholds(image=image, num_layers=num_layers)
        masks = [Image.new("L", image.size, 0) for _ in range(num_layers)]
        mask_access = [mask.load() for mask in masks]
        rgba_pixels = image.load()

        for y in range(height):
            for x in range(width):
                r, g, b, a = rgba_pixels[x, y]
                if a == 0:
                    continue
                luma = int(0.299 * r + 0.587 * g + 0.114 * b)
                layer_index = bisect.bisect_right(thresholds, luma)
                mask_access[layer_index][x, y] = a

        layers: list[Image.Image] = []
        for mask in masks:
            layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            layer.paste(image, (0, 0), mask)
            layers.append(layer)
        return layers

    def _build_luma_thresholds(self, image: Image.Image, num_layers: int) -> list[int]:
        luma_values: list[int] = []
        for r, g, b, a in image.getdata():
            if a > 0:
                luma_values.append(int(0.299 * r + 0.587 * g + 0.114 * b))

        if not luma_values:
            return [0] * max(num_layers - 1, 1)

        luma_values.sort()
        thresholds: list[int] = []
        for i in range(1, num_layers):
            idx = int((i / num_layers) * (len(luma_values) - 1))
            thresholds.append(luma_values[idx])
        return thresholds


def decompose_image(
    image: Image.Image,
    num_layers: int,
    options: dict[str, Any] | None = None,
) -> list[Image.Image]:
    return LayerDecompositionModel.instance().decompose_image(
        image=image,
        num_layers=num_layers,
        options=options,
    )


def runtime_info() -> ModelRuntimeInfo:
    return LayerDecompositionModel.instance().runtime_info()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default
