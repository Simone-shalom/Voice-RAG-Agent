"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  onTranscript: (text: string) => void;
}

const SILENCE_THRESHOLD = 0.015; // RMS amplitude below this counts as silence
const SILENCE_TIMEOUT_MS = 1500; // auto-stop after this much continuous silence once speech has started
const MIN_SPEECH_MS = 300; // ignore silence detection until at least this much speech has been heard

export default function AudioRecorder({ onTranscript }: Props) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vadStatus, setVadStatus] = useState<"idle" | "listening" | "silence">("idle");
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const speechStartedAtRef = useRef<number | null>(null);

  function stopVadLoop() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
    }
    audioCtxRef.current = null;
    analyserRef.current = null;
    silenceStartRef.current = null;
    speechStartedAtRef.current = null;
  }

  function monitorSilence() {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);

    let sumSquares = 0;
    for (let i = 0; i < data.length; i++) {
      const normalized = (data[i] - 128) / 128;
      sumSquares += normalized * normalized;
    }
    const rms = Math.sqrt(sumSquares / data.length);

    const now = performance.now();
    if (speechStartedAtRef.current === null && rms > SILENCE_THRESHOLD) {
      speechStartedAtRef.current = now;
    }
    const speechElapsed = speechStartedAtRef.current !== null ? now - speechStartedAtRef.current : 0;

    if (rms > SILENCE_THRESHOLD) {
      silenceStartRef.current = null;
      setVadStatus("listening");
    } else if (speechStartedAtRef.current !== null && speechElapsed > MIN_SPEECH_MS) {
      if (silenceStartRef.current === null) {
        silenceStartRef.current = now;
      }
      setVadStatus("silence");
      if (now - silenceStartRef.current > SILENCE_TIMEOUT_MS) {
        stop();
        return;
      }
    }

    rafRef.current = requestAnimationFrame(monitorSilence);
  }

  async function start() {
    if (recording || mediaRef.current) return; // guard against double-start (rapid clicks) leaking a stream/context
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
        stopVadLoop();
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        const form = new FormData();
        form.append("audio", blob, "recording.webm");
        try {
          const res = await fetch("/api/voice/stt", { method: "POST", body: form });
          if (!res.ok) throw new Error(await res.text());
          const { text } = await res.json();
          onTranscript(text);
        } catch (err) {
          setError(String(err));
        }
        setVadStatus("idle");
        mediaRef.current = null;
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
      setVadStatus("listening");

      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;
      silenceStartRef.current = null;
      speechStartedAtRef.current = null;
      rafRef.current = requestAnimationFrame(monitorSilence);
    } catch (err) {
      setError(String(err));
    }
  }

  function stop() {
    mediaRef.current?.stop();
    setRecording(false);
  }

  useEffect(() => {
    return () => stopVadLoop();
  }, []);

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        onClick={recording ? stop : start}
        className={`px-6 py-3 rounded-full font-semibold transition-colors ${
          recording
            ? "bg-red-600 hover:bg-red-700 animate-pulse"
            : "bg-indigo-600 hover:bg-indigo-700"
        }`}
      >
        {recording ? "Stop Recording" : "Ask with Voice"}
      </button>
      {recording && (
        <p className="text-xs text-gray-500">
          {vadStatus === "silence" ? "Silence detected — stopping soon…" : "Listening…"}
        </p>
      )}
      {error && <p className="text-red-400 text-sm">{error}</p>}
    </div>
  );
}
