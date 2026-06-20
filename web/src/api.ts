export type Status = { asr: boolean; llm: boolean; model: string; fields: string[] };
export type Word = [string, number];
export type Transcription = { text: string; confidence: number; duration: number; words: Word[] };
export type Patient = { name: string; age: number; gender: string };
export type Fields = Record<string, string>;

export async function getStatus(): Promise<Status> {
  const r = await fetch("/api/status");
  return r.json();
}

export async function transcribe(file: Blob, filename = "audio.wav"): Promise<Transcription> {
  const fd = new FormData();
  fd.append("file", file, filename);
  const r = await fetch("/api/transcribe", { method: "POST", body: fd });
  if (!r.ok) throw new Error("转写失败");
  return r.json();
}

export async function correct(text: string): Promise<string> {
  const r = await fetch("/api/correct", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return (await r.json()).text;
}

/** Stream the structured note. Calls onToken per chunk, onDone with final fields. */
export async function generate(
  req: { transcript: string; patient: Patient; template: string },
  onToken: (t: string) => void,
  onDone: (fields: Fields) => void
): Promise<void> {
  const r = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok || !r.body) throw new Error("生成失败（LLM 是否就绪？）");
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = /event: (\w+)/.exec(chunk)?.[1];
      const dataLine = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      const data = JSON.parse(dataLine.slice(6));
      if (ev === "token") onToken(data.t);
      else if (ev === "done") onDone(data.fields);
    }
  }
}

export async function save(req: {
  record_id?: number | null;
  encounter_id?: number | null;
  patient: Patient;
  transcript: string;
  template: string;
  fields: Fields;
  status: string;
  verified_by?: string;
}): Promise<{ encounter_id: number; record_id: number }> {
  const r = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return r.json();
}

export async function exportNote(
  fields: Fields,
  patient: Patient,
  template: string,
  fmt: "pdf" | "txt" | "md"
): Promise<Blob> {
  const r = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fields, patient, template, fmt }),
  });
  return r.blob();
}

export async function listEncounters() {
  return (await fetch("/api/encounters")).json();
}
export async function getEncounter(id: number) {
  return (await fetch(`/api/encounters/${id}`)).json();
}
