#!/usr/bin/env python3
"""
CardioVoice - Streamlit MVP (local-first)
=========================================

A single-Mac UI for the doctor: record a visit from the local microphone (or
upload an audio file), transcribe with MedASR, generate a structured outpatient
note with a local LLM, edit it, and save / verify it to a local SQLite store.

Everything runs on the machine; no audio or notes leave it.

Run:
    conda activate medgemma
    ollama serve                     # in another terminal
    streamlit run app.py
"""

import queue
import sys
import tempfile
import threading
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from unified_model_manager import (  # noqa: E402
    UnifiedModelManager,
    create_patient_info,
    ModelStatus,
)
from store import RecordStore, NOTE_FIELDS  # noqa: E402
from audio_capture import MicSource, list_input_devices  # noqa: E402
from vad_segmenter import VADSegmenter  # noqa: E402

NOTE_LABELS = {
    "chief_complaint": "主诉 / Chief Complaint",
    "present_history": "现病史 / HPI",
    "past_history": "既往史 / PMH",
    "cardiovascular_exam": "查体 / Exam",
    "ecg_findings": "ECG",
    "assessment": "评估诊断 / Assessment",
    "plan": "处理计划 / Plan",
}

_PLACEHOLDERS = {
    "{period}": ".",
    "{comma}": ",",
    "{colon}": ":",
    "{new paragraph}": "\n",
    "{open_bracket}": "[",
    "{close_bracket}": "]",
    "</s>": "",
}


def clean_transcript(text: str) -> str:
    """Make MedASR's literal placeholder output human-readable."""
    for k, v in _PLACEHOLDERS.items():
        text = text.replace(k, v)
    return " ".join(text.split())


# --------------------------------------------------------------------- models
@st.cache_resource(show_spinner="加载模型中 (MedASR + Ollama)...")
def get_manager() -> UnifiedModelManager:
    mgr = UnifiedModelManager()
    mgr.load_all(verbose=False)
    return mgr


@st.cache_resource
def get_store() -> RecordStore:
    return RecordStore()


# ------------------------------------------------------------------ recorder
class MicRecorder:
    """Mic -> VAD -> MedASR worker, accumulating transcript text. Lives in
    st.session_state across reruns."""

    def __init__(self, manager: UnifiedModelManager, device=None):
        self.manager = manager
        self.seg_q: "queue.Queue" = queue.Queue(maxsize=64)
        self._segments = []
        self._lock = threading.Lock()
        self.segmenter = VADSegmenter(on_segment=self._on_segment)
        self.source = MicSource(device=device)
        self._asr_thread = None
        self.running = False

    def _on_segment(self, seg):
        try:
            self.seg_q.put_nowait(seg)
        except queue.Full:
            pass

    def _asr_loop(self):
        while True:
            item = self.seg_q.get()
            if item is None:
                break
            try:
                r = self.manager.medasr.transcribe_array(item, sample_rate=16000)
                t = r.text.strip()
                if t:
                    with self._lock:
                        self._segments.append(t)
            except Exception:
                pass

    def start(self):
        self.running = True
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._asr_thread.start()
        self.source.start(on_frame=self.segmenter.accept)

    def stop(self):
        try:
            self.source.stop()
        except Exception:
            pass
        self.segmenter.flush()
        self.seg_q.put(None)
        if self._asr_thread:
            self._asr_thread.join(timeout=5)
        self.running = False

    def transcript(self) -> str:
        with self._lock:
            return clean_transcript(" ".join(self._segments))


# ------------------------------------------------------------------ helpers
def _reset_note_state():
    for k in ("transcript", "note_fields", "encounter_id", "record_id", "model_used"):
        st.session_state.pop(k, None)


def _generate_note(manager, transcript, patient, template):
    record = manager.generate_record(
        transcription=transcript, patient_info=patient, template=template
    )
    return {k: getattr(record, k, "") for k in NOTE_FIELDS}, record.raw_response


# ------------------------------------------------------------------- sidebar
def render_sidebar(manager):
    st.sidebar.header("🏥 CardioVoice")
    st.sidebar.caption("本地优先 · 数据不出本机")

    asr_ok = manager.medasr.is_available()
    llm_ok = manager.medgemma.is_available()
    st.sidebar.write(f"MedASR: {'🟢' if asr_ok else '🔴'}")
    st.sidebar.write(
        f"LLM ({manager.medgemma.model_name}): {'🟢' if llm_ok else '🔴 需启动 ollama serve'}"
    )

    st.sidebar.divider()
    st.sidebar.subheader("患者信息")
    name = st.sidebar.text_input("姓名", value="", key="pt_name")
    col1, col2 = st.sidebar.columns(2)
    age = col1.number_input("年龄", min_value=0, max_value=120, value=58, key="pt_age")
    gender = col2.selectbox("性别", ["male", "female", "unknown"], key="pt_gender")

    st.sidebar.subheader("生成设置")
    template = st.sidebar.selectbox(
        "病历模板", ["cardiology", "general"], key="template"
    )
    return {
        "patient": {"name": name or "Patient", "age": int(age), "gender": gender},
        "template": template,
    }


# -------------------------------------------------------------------- inputs
def render_input(manager, device_map):
    st.subheader("1 · 采集 / 转写")
    mode = st.radio("输入方式", ["🎤 麦克风录音", "📁 上传音频"], horizontal=True)

    if mode == "🎤 麦克风录音":
        device_label = st.selectbox("输入设备", list(device_map.keys()))
        device = device_map[device_label]
        rec = st.session_state.get("recorder")

        c1, c2 = st.columns(2)
        if not (rec and rec.running):
            if c1.button("● 开始录音", type="primary", use_container_width=True):
                recorder = MicRecorder(manager, device=device)
                recorder.start()
                st.session_state["recorder"] = recorder
                st.rerun()
        else:
            if c2.button("■ 停止并转写", type="primary", use_container_width=True):
                rec.stop()
                st.session_state["transcript"] = rec.transcript()
                st.session_state.pop("recorder", None)
                st.rerun()

        rec = st.session_state.get("recorder")
        if rec and rec.running:
            st.info("🔴 录音中… 说话停顿处会自动分句转写")

            @st.fragment(run_every=2)
            def _live():
                st.text_area(
                    "实时转写", value=rec.transcript(), height=160, disabled=True
                )

            _live()

    else:
        up = st.file_uploader("音频文件 (wav/mp3/m4a/flac)", type=["wav", "mp3", "m4a", "flac"])
        if up and st.button("转写", type="primary"):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(up.name).suffix
            ) as f:
                f.write(up.getbuffer())
                tmp = f.name
            with st.spinner("MedASR 转写中…"):
                result = manager.transcribe(tmp)
            st.session_state["transcript"] = clean_transcript(result.text)
            st.rerun()


# --------------------------------------------------------------------- note
def render_note(manager, cfg, store):
    transcript = st.session_state.get("transcript")
    if not transcript:
        return

    st.subheader("2 · 转写文本")
    st.session_state["transcript"] = st.text_area(
        "可手动修正后再生成", value=transcript, height=140
    )

    if st.button("🧠 生成病历", type="primary"):
        if not manager.medgemma.is_available():
            st.error("LLM 不可用，请先运行 `ollama serve`")
            return
        patient = create_patient_info(**cfg["patient"])
        with st.spinner(f"{manager.medgemma.model_name} 生成中…（约 30s）"):
            fields, raw = _generate_note(
                manager, st.session_state["transcript"], patient, cfg["template"]
            )
        enc_id = store.create_encounter(
            patient=cfg["patient"],
            transcript=st.session_state["transcript"],
            template=cfg["template"],
        )
        rec_id = store.save_record(
            enc_id, fields, raw_response=raw, model=manager.medgemma.model_name
        )
        st.session_state.update(
            note_fields=fields,
            encounter_id=enc_id,
            record_id=rec_id,
            model_used=manager.medgemma.model_name,
        )
        st.rerun()

    fields = st.session_state.get("note_fields")
    if not fields:
        return

    st.subheader("3 · 病历（可编辑）")
    st.caption(f"模型: {st.session_state.get('model_used','')}  ·  草稿需医生核验后保存")
    edited = {}
    for key in NOTE_FIELDS:
        edited[key] = st.text_area(
            NOTE_LABELS[key], value=fields.get(key, ""), height=90, key=f"f_{key}"
        )

    c1, c2, c3 = st.columns(3)
    if c1.button("💾 保存草稿", use_container_width=True):
        store.save_record(
            st.session_state["encounter_id"],
            edited,
            status="draft",
            record_id=st.session_state["record_id"],
        )
        st.session_state["note_fields"] = edited
        st.success("已保存草稿")
    verifier = c3.text_input("核验医生", value=cfg["patient"].get("name") and "" or "")
    if c2.button("✅ 核验并保存", type="primary", use_container_width=True):
        store.save_record(
            st.session_state["encounter_id"],
            edited,
            status="verified",
            record_id=st.session_state["record_id"],
        )
        store.verify_record(st.session_state["record_id"], verifier or "physician")
        st.session_state["note_fields"] = edited
        st.success("已核验并保存 ✅")


# ------------------------------------------------------------------ history
def render_history(store):
    st.subheader("历史记录")
    encs = store.list_encounters()
    if not encs:
        st.caption("暂无记录")
        return
    for e in encs:
        status = e.get("record_status") or "—"
        with st.expander(
            f"#{e['id']} · {e['patient_name']} · {e['created_at']} · [{status}]"
        ):
            rec = store.get_record_for_encounter(e["id"])
            st.caption("转写")
            st.write(e.get("transcript") or "（无）")
            if rec:
                for key in NOTE_FIELDS:
                    val = rec["fields"].get(key)
                    if val:
                        st.markdown(f"**{NOTE_LABELS[key]}**: {val}")


# ---------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="CardioVoice", page_icon="🏥", layout="centered")
    manager = get_manager()
    store = get_store()
    cfg = render_sidebar(manager)

    try:
        devices = list_input_devices()
        device_map = {f"[{d['index']}] {d['name']}": d["index"] for d in devices}
    except Exception:
        device_map = {"默认设备": None}

    tab_new, tab_hist = st.tabs(["📝 新建记录", "📚 历史"])
    with tab_new:
        c1, c2 = st.columns([3, 1])
        c1.title("门诊记录")
        if c2.button("🔄 新建"):
            _reset_note_state()
            st.rerun()
        render_input(manager, device_map)
        render_note(manager, cfg, store)
    with tab_hist:
        render_history(store)


if __name__ == "__main__":
    main()
