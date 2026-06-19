#!/usr/bin/env python3
"""
CardioVoice - Streamlit MVP (local-first)
=========================================

Doctor-facing UI for a single Mac. Record the visit in the browser (or upload
audio), transcribe with MedASR, generate a structured outpatient note with a
local LLM, edit it, and save / verify to a local SQLite store. Nothing leaves
the machine.

Run:
    conda activate medgemma
    ollama serve                     # in another terminal
    streamlit run app.py
"""

import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Quiet the console (httpx logs every Ollama poll).
for _n in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_n).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from unified_model_manager import UnifiedModelManager, create_patient_info  # noqa: E402
from store import RecordStore, NOTE_FIELDS  # noqa: E402

NOTE_LABELS = {
    "chief_complaint": "主诉 · Chief Complaint",
    "present_history": "现病史 · HPI",
    "past_history": "既往史 · PMH",
    "cardiovascular_exam": "查体 · Exam",
    "ecg_findings": "心电图 · ECG",
    "assessment": "评估 · Assessment",
    "plan": "计划 · Plan",
}

_PLACEHOLDERS = {
    "{period}": ". ", "{comma}": ", ", "{colon}": ": ",
    "{new paragraph}": "\n", "{open_bracket}": "[", "{close_bracket}": "]", "</s>": "",
}


def clean_transcript(text: str) -> str:
    for k, v in _PLACEHOLDERS.items():
        text = text.replace(k, v)
    return " ".join(text.split())


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="正在加载模型 (MedASR + Ollama)…")
def get_manager() -> UnifiedModelManager:
    mgr = UnifiedModelManager()
    mgr.load_all(verbose=False)
    return mgr


@st.cache_resource
def get_store() -> RecordStore:
    return RecordStore()


# ---------------------------------------------------------------------- style
CSS = """
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {max-width: 920px; padding-top: 1.2rem; padding-bottom: 3rem;}

.cv-hero {
  background: linear-gradient(135deg,#0E7C7B 0%,#12A3A1 100%);
  border-radius: 16px; padding: 20px 24px; color: #fff; margin-bottom: 18px;
}
.cv-hero h1 {font-size: 1.5rem; margin: 0; font-weight: 700; color:#fff;}
.cv-hero p {margin: 4px 0 0; opacity: .9; font-size: .9rem;}

.cv-pills {margin-top: 12px;}
.cv-pill {
  display:inline-flex; align-items:center; gap:6px; background: rgba(255,255,255,.16);
  border-radius: 999px; padding: 4px 12px; font-size: .8rem; margin-right: 8px;
}
.cv-dot {width:8px; height:8px; border-radius:50%; display:inline-block;}
.cv-on {background:#42E2A2;} .cv-off {background:#FF6B6B;}

.cv-step {display:flex; align-items:center; gap:10px; margin: 4px 0 10px;}
.cv-num {
  width:26px; height:26px; border-radius:50%; background:#0E7C7B; color:#fff;
  display:flex; align-items:center; justify-content:center; font-size:.85rem; font-weight:700;
}
.cv-step .t {font-size:1.05rem; font-weight:700; color:#13293D;}

div[data-testid="stTextArea"] textarea {font-size:.92rem; line-height:1.5;}
.stButton button {border-radius: 10px; font-weight: 600;}
.cv-tag {font-size:.78rem; color:#5B7083;}
</style>
"""


def hero(asr_ok: bool, llm_ok: bool, model_name: str):
    def pill(ok, label):
        cls = "cv-on" if ok else "cv-off"
        return f'<span class="cv-pill"><span class="cv-dot {cls}"></span>{label}</span>'

    st.markdown(
        f"""
        <div class="cv-hero">
          <h1>🫀 CardioVoice</h1>
          <p>心内科门诊语音病历 · 本地运行，数据不出本机</p>
          <div class="cv-pills">
            {pill(asr_ok, "MedASR 转写")}
            {pill(llm_ok, f"LLM · {model_name}" + ("" if llm_ok else " (需 ollama serve)"))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step(num: int, title: str):
    st.markdown(
        f'<div class="cv-step"><div class="cv-num">{num}</div>'
        f'<div class="t">{title}</div></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------- actions
def _transcribe_bytes(manager, data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    result = manager.transcribe(path)
    return clean_transcript(result.text)


def _reset():
    for k in ("transcript", "note_fields", "encounter_id", "record_id",
              "model_used", "audio_sig"):
        st.session_state.pop(k, None)


# --------------------------------------------------------------------- panels
def patient_panel():
    with st.container(border=True):
        step(1, "患者信息")
        c1, c2, c3, c4 = st.columns([3, 1.2, 1.4, 1.8])
        name = c1.text_input("姓名", key="pt_name", placeholder="可留空")
        age = c2.number_input("年龄", 0, 120, 58, key="pt_age")
        gender = c3.selectbox("性别", ["male", "female", "unknown"], key="pt_gender")
        template = c4.selectbox("模板", ["cardiology", "general"], key="template")
    return {
        "patient": {"name": name or "Patient", "age": int(age), "gender": gender},
        "template": template,
    }


def audio_panel(manager):
    with st.container(border=True):
        step(2, "采集 / 转写")
        tab_rec, tab_up = st.tabs(["🎤 录音", "📁 上传音频"])

        data, suffix = None, ".wav"
        with tab_rec:
            st.caption("点击麦克风开始/停止录音，松手后自动转写")
            rec = st.audio_input("录音", label_visibility="collapsed")
            if rec is not None:
                data, suffix = rec.getvalue(), ".wav"
        with tab_up:
            up = st.file_uploader(
                "音频文件", type=["wav", "mp3", "m4a", "flac"],
                label_visibility="collapsed",
            )
            if up is not None:
                data, suffix = up.getbuffer().tobytes(), Path(up.name).suffix

        # Transcribe only when the audio actually changes.
        if data:
            sig = hash(bytes(data))
            if st.session_state.get("audio_sig") != sig:
                with st.spinner("MedASR 转写中…"):
                    try:
                        st.session_state["transcript"] = _transcribe_bytes(
                            manager, bytes(data), suffix
                        )
                        st.session_state["audio_sig"] = sig
                        st.session_state.pop("note_fields", None)
                    except Exception as e:
                        st.error(f"转写失败: {e}")

        transcript = st.session_state.get("transcript")
        if transcript is not None:
            st.markdown('<span class="cv-tag">转写结果（可手动修正）</span>',
                        unsafe_allow_html=True)
            st.session_state["transcript"] = st.text_area(
                "transcript", value=transcript, height=150,
                label_visibility="collapsed",
            )
            if not transcript.strip():
                st.warning("没有识别到语音内容。请确认麦克风权限，并贴近说话。")


def note_panel(manager, cfg, store):
    if not st.session_state.get("transcript", "").strip():
        return
    with st.container(border=True):
        step(3, "生成结构化病历")
        llm_ok = manager.medgemma.is_available()
        if st.button("🧠 生成病历", type="primary", disabled=not llm_ok):
            patient = create_patient_info(**cfg["patient"])
            with st.spinner(f"{manager.medgemma.model_name} 生成中…（约 30 秒）"):
                record = manager.generate_record(
                    transcription=st.session_state["transcript"],
                    patient_info=patient,
                    template=cfg["template"],
                )
            fields = {k: getattr(record, k, "") for k in NOTE_FIELDS}
            enc_id = store.create_encounter(
                patient=cfg["patient"],
                transcript=st.session_state["transcript"],
                template=cfg["template"],
            )
            rec_id = store.save_record(
                enc_id, fields, raw_response=record.raw_response,
                model=manager.medgemma.model_name,
            )
            st.session_state.update(
                note_fields=fields, encounter_id=enc_id, record_id=rec_id,
                model_used=manager.medgemma.model_name,
            )
            st.rerun()
        if not llm_ok:
            st.info("LLM 未就绪：请在另一个终端运行 `ollama serve`。")

    fields = st.session_state.get("note_fields")
    if not fields:
        return

    with st.container(border=True):
        step(4, "审阅与编辑")
        st.markdown(
            f'<span class="cv-tag">模型 {st.session_state.get("model_used","")} '
            "生成 · 医生核验后保存</span>", unsafe_allow_html=True)
        edited = {}
        for key in NOTE_FIELDS:
            edited[key] = st.text_area(
                NOTE_LABELS[key], value=fields.get(key, ""), height=84, key=f"f_{key}"
            )
        st.divider()
        c1, c2, c3 = st.columns([1.4, 1.4, 2])
        verifier = c3.text_input("核验医生", placeholder="签名", label_visibility="collapsed")
        if c1.button("💾 保存草稿", use_container_width=True):
            store.save_record(st.session_state["encounter_id"], edited,
                              status="draft", record_id=st.session_state["record_id"])
            st.session_state["note_fields"] = edited
            st.toast("已保存草稿", icon="💾")
        if c2.button("✅ 核验并保存", type="primary", use_container_width=True):
            store.save_record(st.session_state["encounter_id"], edited,
                              status="verified", record_id=st.session_state["record_id"])
            store.verify_record(st.session_state["record_id"], verifier or "physician")
            st.session_state["note_fields"] = edited
            st.toast("已核验并保存", icon="✅")


def history_panel(store):
    encs = store.list_encounters()
    if not encs:
        st.caption("暂无记录")
        return
    for e in encs:
        status = (e.get("record_status") or "—")
        badge = {"verified": "✅", "draft": "📝"}.get(status, "·")
        with st.expander(f"{badge}  #{e['id']} · {e['patient_name']} · {e['created_at']}"):
            rec = store.get_record_for_encounter(e["id"])
            if rec:
                for key in NOTE_FIELDS:
                    if rec["fields"].get(key):
                        st.markdown(f"**{NOTE_LABELS[key]}** — {rec['fields'][key]}")
            st.markdown('<span class="cv-tag">原始转写</span>', unsafe_allow_html=True)
            st.caption(e.get("transcript") or "（无）")


# ---------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="CardioVoice", page_icon="🫀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    manager = get_manager()
    store = get_store()
    asr_ok = manager.medasr.is_available()
    llm_ok = manager.medgemma.is_available()

    hero(asr_ok, llm_ok, manager.medgemma.model_name)

    top = st.columns([6, 1])
    if top[1].button("🔄 新建", use_container_width=True):
        _reset()
        st.rerun()

    tab_new, tab_hist = st.tabs(["📝 新建门诊记录", "📚 历史记录"])
    with tab_new:
        cfg = patient_panel()
        audio_panel(manager)
        note_panel(manager, cfg, store)
    with tab_hist:
        history_panel(store)


if __name__ == "__main__":
    main()
