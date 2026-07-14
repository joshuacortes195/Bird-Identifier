"""Optional internal Gradio demo (`make demo`) for quick manual testing.

This is a developer convenience only — the real user surface is the React app (Phase 10)
talking to the FastAPI service. Gradio is imported lazily so it isn't a hard serve
dependency. Uses the same :class:`~wildlife.serve.predictor.Predictor` the API uses.
"""

from __future__ import annotations

from wildlife.serve.config import ServeConfig
from wildlife.serve.predictor import build_predictor


def build_demo(config: ServeConfig | None = None):
    import gradio as gr

    cfg = config or ServeConfig.from_env()
    predictor = build_predictor(
        model_path=cfg.model_path,
        taxonomy_path=cfg.taxonomy_path,
        preprocess_cfg=cfg.preprocess,
        allow_stub=cfg.allow_stub,
    )

    def classify(image):
        if image is None:
            return {}
        preds = predictor.predict(image, top_k=cfg.top_k)
        # gr.Label expects {label: confidence}.
        return {p.common_name: p.confidence for p in preds}

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Bird photo"),
        outputs=gr.Label(num_top_classes=cfg.top_k, label="Top species"),
        title="Wildlife Classifier — internal demo",
        description=f"Backend: {predictor.info.backend} · {predictor.info.num_classes} classes",
        allow_flagging="never",
    )


def main() -> None:
    build_demo().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
