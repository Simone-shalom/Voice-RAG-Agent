"use client";

import { useRef, useState } from "react";

interface Props {
  onTranscript: (text: string) => void;
}

export default function AudioRecorder({ onTranscript }: Props) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
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
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
    } catch (err) {
      setError(String(err));
    }
  }

  function stop() {
    mediaRef.current?.stop();
    setRecording(false);
  }

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
      {error && <p className="text-red-400 text-sm">{error}</p>}
    </div>
  );
}
