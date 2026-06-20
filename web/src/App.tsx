import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { Fields, Patient, Word } from "./api";
import { WavRecorder } from "./recorder";
import { Ecg } from "./Ecg";

const LABELS: Record<string, string> = {
  chief_complaint: "主诉 · Chief Complaint",
  present_history: "现病史 · HPI",
  past_history: "既往史 · PMH",
  cardiovascular_exam: "查体 · Exam",
  ecg_findings: "心电图 · ECG",
  assessment: "评估 · Assessment",
  plan: "计划 · Plan",
};
const FIELD_ORDER = Object.keys(LABELS);
const LOW_CONF = 0.6;
const PH: Record<string, string> = {
  "{period}": ". ", "{comma}": ", ", "{colon}": ": ",
  "{new paragraph}": "\n", "{open_bracket}": "[", "{close_bracket}": "]", "</s>": "",
};

function cleanWord(w: string): string {
  return PH[w] !== undefined ? PH[w] : w;
}

function Led({ on, label }: { on: boolean; label: string }) {
  return (
    <span className="led">
      <span className={"dot " + (on ? "on" : "off")} />
      {label}
    </span>
  );
}

function Step({ n, label, sub }: { n: string; label: string; sub: string }) {
  return (
    <div className="step">
      <span className="lead">{n}</span>
      <span className="label">
        {label} <small>· {sub}</small>
      </span>
    </div>
  );
}

function ConfidenceView({ words }: { words: Word[] }) {
  if (!words.length) return null;
  const flagged = words.filter(([w, c]) => c < LOW_CONF && PH[w] === undefined).length;
  return (
    <div className="conf-box">
      <div className="conf-head">
        {flagged > 0
          ? `${flagged} 处低置信片段（虚线标注）— 请重点核对`
          : "全部高置信 — 无需特别核对"}
      </div>
      <div className="conf-text">
        {words.map(([w, c], i) => {
          const disp = cleanWord(w);
          if (PH[w] !== undefined) return <span key={i}>{disp}</span>;
          const low = c < LOW_CONF;
          return (
            <span key={i} className={low ? "low" : ""} title={`置信度 ${Math.round(c * 100)}%`}>
              {disp}{" "}
            </span>
          );
        })}
      </div>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState<api.Status | null>(null);
  const [tab, setTab] = useState<"new" | "history">("new");

  const [patient, setPatient] = useState<Patient>({ name: "", age: 58, gender: "male" });
  const [template, setTemplate] = useState("cardiology");

  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState<string>("");
  const recRef = useRef<WavRecorder | null>(null);

  const [transcript, setTranscript] = useState<string>("");
  const [words, setWords] = useState<Word[]>([]);
  const [conf, setConf] = useState(0);

  const [fields, setFields] = useState<Fields | null>(null);
  const [streamRaw, setStreamRaw] = useState("");
  const [genLive, setGenLive] = useState(false);
  const [ids, setIds] = useState<{ e?: number; r?: number }>({});

  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    api.getStatus().then(setStatus).catch(() => {});
  }, []);

  function reset() {
    setTranscript(""); setWords([]); setConf(0); setFields(null);
    setStreamRaw(""); setIds({});
  }

  async function onAudio(blob: Blob, name = "audio.wav") {
    setBusy("MedASR 转写中…");
    try {
      const t = await api.transcribe(blob, name);
      setTranscript(t.text); setWords(t.words); setConf(t.confidence); setFields(null);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setBusy("");
    }
  }

  async function toggleRecord() {
    if (!recording) {
      recRef.current = new WavRecorder();
      await recRef.current.start();
      setRecording(true);
    } else {
      setRecording(false);
      const blob = await recRef.current!.stop();
      await onAudio(blob, "recording.wav");
    }
  }

  async function runCorrect() {
    setBusy("术语纠错中…");
    try {
      setTranscript(await api.correct(transcript));
      setWords([]);
    } finally {
      setBusy("");
    }
  }

  async function runGenerate() {
    setGenLive(true); setStreamRaw(""); setFields(null);
    try {
      await api.generate({ transcript, patient, template },
        (t) => setStreamRaw((s) => s + t),
        async (f) => {
          setFields(f);
          const res = await api.save({
            patient, transcript, template, fields: f, status: "draft",
          });
          setIds({ e: res.encounter_id, r: res.record_id });
        });
    } catch (e: any) {
      alert(e.message);
    } finally {
      setGenLive(false);
    }
  }

  async function persist(statusVal: string, verifier?: string) {
    if (!fields) return;
    const res = await api.save({
      record_id: ids.r, encounter_id: ids.e, patient, transcript, template,
      fields, status: statusVal, verified_by: verifier,
    });
    setIds({ e: res.encounter_id, r: res.record_id });
    alert(statusVal === "verified" ? "已核验并保存 ✅" : "已保存草稿 💾");
  }

  async function download(fmt: "pdf" | "txt" | "md") {
    if (!fields) return;
    const blob = await api.exportNote(fields, patient, template, fmt);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `note_${patient.name || "patient"}.${fmt}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const emptyCount = fields ? FIELD_ORDER.filter((k) => !(fields[k] || "").trim()).length : 0;
  const trace: "idle" | "recording" | "busy" =
    recording ? "recording" : busy || genLive ? "busy" : "idle";

  return (
    <div className="app">
      <header className="bar">
        <div className="brand">
          <span className="mark">Cardio<b>Voice</b></span>
          <span className="sub">ambient cardiology notes · local</span>
        </div>
        <Ecg state={trace} />
        <div className="leds">
          <Led on={!!status?.asr} label="MedASR" />
          <Led on={!!status?.llm} label={status?.model ?? "LLM"} />
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === "new" ? "active" : ""} onClick={() => setTab("new")}>
          新建记录
        </button>
        <button className={tab === "history" ? "active" : ""}
          onClick={() => { setTab("history"); api.listEncounters().then(setHistory); }}>
          历史
        </button>
        <button className="ghost" onClick={reset}>清空重来</button>
      </nav>

      {tab === "new" && (
        <main>
          <section className="card">
            <Step n="01" label="患者信息" sub="Patient" />
            <div className="row">
              <label className="grow"><span className="flabel">姓名 Name</span>
                <input value={patient.name}
                  onChange={(e) => setPatient({ ...patient, name: e.target.value })}
                  placeholder="可留空" />
              </label>
              <label><span className="flabel">年龄 Age</span>
                <input type="number" value={patient.age}
                  onChange={(e) => setPatient({ ...patient, age: +e.target.value })} />
              </label>
              <label><span className="flabel">性别 Sex</span>
                <select value={patient.gender}
                  onChange={(e) => setPatient({ ...patient, gender: e.target.value })}>
                  <option value="male">male</option>
                  <option value="female">female</option>
                  <option value="unknown">unknown</option>
                </select>
              </label>
              <label><span className="flabel">模板 Template</span>
                <select value={template} onChange={(e) => setTemplate(e.target.value)}>
                  <option value="cardiology">cardiology</option>
                  <option value="general">general</option>
                </select>
              </label>
            </div>
          </section>

          <section className="card">
            <Step n="02" label="采集与转写" sub="Capture" />
            <div className="row">
              <button className={"rec " + (recording ? "on" : "")} onClick={toggleRecord}>
                {recording ? "停止并转写" : "开始录音"}
              </button>
              <label className="upload">
                上传音频
                <input type="file" accept="audio/*" hidden
                  onChange={(e) => e.target.files?.[0] && onAudio(e.target.files[0], e.target.files[0].name)} />
              </label>
              {busy && <span className="busy">{busy}</span>}
              {recording && <span className="busy rec">录音中</span>}
            </div>

            {words.length > 0 && (
              <div className="metric">
                <span className="n">{Math.round(conf * 100)}%</span>
                <span className="cap">整体置信度</span>
                <span className="gauge"><i style={{ width: `${Math.round(conf * 100)}%` }} /></span>
              </div>
            )}
            <ConfidenceView words={words} />

            {transcript !== "" && (
              <>
                <div className="tag">转写结果（可手动修正后再生成）</div>
                <textarea value={transcript} onChange={(e) => setTranscript(e.target.value)} rows={5} />
                <button className="link" disabled={!status?.llm} onClick={runCorrect}>
                  🩺 术语纠错（LLM 保守修正医学术语/药名）
                </button>
              </>
            )}
          </section>

          {transcript.trim() && (
            <section className="card">
              <Step n="03" label="生成结构化病历" sub="Note" />
              <button className="primary" disabled={!status?.llm || genLive} onClick={runGenerate}>
                {genLive ? "生成中…（实时）" : "生成病历"}
              </button>
              {!status?.llm && <p className="hint">LLM 未就绪：请运行 <code>ollama serve</code></p>}
              {genLive && <pre className="stream">{streamRaw || "…"}</pre>}
            </section>
          )}

          {fields && (
            <section className="card">
              <Step n="04" label="审阅与编辑" sub="Review" />
              <div className="tag">
                仅依据转写内容 · 未提及的查体/检查留空（不编造）· 医生核验后保存
              </div>
              {emptyCount > 0 && (
                <p className="hint">ℹ️ {emptyCount} 个字段为空 = 转写中未提及，请补充（留空是有意为之）</p>
              )}
              {FIELD_ORDER.map((k) => (
                <label key={k} className="field"><span className="flabel">{LABELS[k]}</span>
                  <textarea value={fields[k] || ""} rows={3}
                    placeholder={!(fields[k] || "").trim() ? "转写中未提及 — 待补充" : ""}
                    onChange={(e) => setFields({ ...fields, [k]: e.target.value })} />
                </label>
              ))}
              <div className="row actions">
                <button onClick={() => persist("draft")}>💾 保存草稿</button>
                <button className="primary"
                  onClick={() => persist("verified", prompt("核验医生签名：") || "physician")}>
                  ✅ 核验并保存
                </button>
                <span className="spacer" />
                <button className="ghost" onClick={() => download("txt")}>⬇️ 文本</button>
                <button className="ghost" onClick={() => download("md")}>⬇️ MD</button>
                <button className="ghost" onClick={() => download("pdf")}>⬇️ PDF</button>
              </div>
            </section>
          )}
        </main>
      )}

      {tab === "history" && (
        <main>
          <section className="card">
            <Step n="HX" label="历史记录" sub="History" />
            {history.length === 0 && <p className="hint">暂无记录</p>}
            {history.map((e) => (
              <details key={e.id} className="hist">
                <summary>
                  {(e.record_status === "verified" ? "✅" : "📝")} #{e.id} ·{" "}
                  {e.patient_name} · {e.created_at}
                </summary>
                <p className="tag">转写</p>
                <p className="muted">{e.transcript || "（无）"}</p>
              </details>
            ))}
          </section>
        </main>
      )}
    </div>
  );
}
