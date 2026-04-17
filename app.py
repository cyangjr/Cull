from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.session import SessionManager


def _thumb_path_cache_key(path: str) -> str:
    return f"thumb::{path}"


@st.cache_data(show_spinner=False)
def load_thumbnail(path: str, max_size: int = 256) -> bytes | None:
    try:
        from PIL import Image, ImageOps

        p = Path(path)
        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_size, max_size))
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Cull", layout="wide")
    st.title("Cull")
    st.caption("AI-powered photo selection (Phase A MVP)")

    with st.sidebar:
        st.subheader("Input")
        folder_path = st.text_input("Folder path", value="")
        sharpness_threshold = st.number_input("Sharpness gate threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

        st.subheader("Results")
        min_score = st.slider("Min final score", 0.0, 1.0, 0.0, 0.01)
        show_failed = st.checkbox("Show gate-failed", value=True)

        st.subheader("Load previous results")
        uploaded = st.file_uploader("Load exported JSON", type=["json"])

    session = st.session_state.get("session_manager")
    if session is None:
        session = SessionManager()
        st.session_state["session_manager"] = session

    with st.sidebar:
        _device = session.pipeline.device
        _device_labels = {"cuda": "GPU — NVIDIA CUDA", "mps": "GPU — Apple Silicon", "cpu": "CPU"}
        st.caption(f"Device: {_device_labels.get(_device, _device)}")

    # Allow quick tuning without editing YAML yet.
    session.pipeline.config.sharpness_gate_threshold = float(sharpness_threshold)

    if uploaded is not None:
        tmp_path = Path(st.session_state.get("loaded_json_path", "loaded_results.json"))
        tmp_path.write_bytes(uploaded.getvalue())
        records = SessionManager.load_json(str(tmp_path))
        st.session_state["records"] = records
        st.info("Loaded records from JSON (pixel arrays are not included; thumbnails are lazy-loaded from path).")

    col_run, col_export = st.columns([1, 1])
    with col_run:
        if st.button("Run cull", type="primary", disabled=not folder_path):
            progress = st.progress(0.0)
            status = st.empty()

            def on_progress(frac: float, msg: str) -> None:
                progress.progress(max(0.0, min(1.0, float(frac))))
                status.text(msg)

            cleaned = folder_path.strip().strip('"').strip("'").strip()
            try:
                session.start(cleaned, progress_callback=on_progress)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

            st.session_state["records"] = session.records
            status.text("Done")

    with col_export:
        if st.button("Export kept to JSON"):
            out = Path("export.json")
            session.export_json(str(out))
            st.success(f"Exported to {out.resolve()}")

    records = st.session_state.get("records") or []
    if not records:
        st.stop()

    # Filter records
    filtered = []
    for r in records:
        if (r.final_score or 0.0) < min_score:
            continue
        if not show_failed and not r.passed_gate:
            continue
        filtered.append(r)

    st.subheader(f"Results ({len(filtered)} shown)")

    df = pd.DataFrame(
        [
            {
                "filename": r.filename,
                "final_score": r.final_score,
                "passed_gate": r.passed_gate,
                "has_faces": r.has_faces,
                "scene_type": r.scene_type,
                "is_duplicate": r.is_duplicate,
                "motion_blur": r.motion_blur_detected,
                "aesthetic": r.aesthetic_score,
                "tags": ",".join(r.composition_tags or []),
                "sharpness": r.sharpness_score,
                "exposure": r.exposure_score,
                "white_balance": r.white_balance_score,
                "timestamp": (r.exif or {}).get("timestamp"),
                "iso": (r.exif or {}).get("iso"),
                "path": r.path,
            }
            for r in filtered
        ]
    ).sort_values(by=["final_score"], ascending=False, na_position="last")

    st.dataframe(df, width="stretch", hide_index=True)

    st.subheader("Thumbnails")
    top10 = df.head(10).to_dict(orient="records")
    bottom10 = df.tail(10).sort_values(by=["final_score"], ascending=True, na_position="last").to_dict(orient="records")

    col_top, col_bottom = st.columns(2)

    with col_top:
        st.markdown("**Top 10**")
        cols = st.columns(2)
        for i, r in enumerate(top10):
            b = load_thumbnail(r["path"])
            with cols[i % 2]:
                if b is not None:
                    st.image(b, caption=f"{r['filename']}\n{(r.get('final_score') or 0.0):.3f}")
                else:
                    st.write(r["filename"])

    with col_bottom:
        st.markdown("**Bottom 10**")
        cols = st.columns(2)
        for i, r in enumerate(bottom10):
            b = load_thumbnail(r["path"])
            with cols[i % 2]:
                if b is not None:
                    st.image(b, caption=f"{r['filename']}\n{(r.get('final_score') or 0.0):.3f}")
                else:
                    st.write(r["filename"])


if __name__ == "__main__":
    main()

