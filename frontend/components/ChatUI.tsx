"use client";

import { useRef, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import SourcePlayer from "./SourcePlayer";

interface Source {
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
}

interface Message {
  role: "user" | "assistant";
  text: string;
  audioUrl?: string;
  sources?: Source[];
}

export default function ChatUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const threadId = useRef(crypto.randomUUID());

  async function sendMessage(text: string) {
    if (!text.trim()) return;
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text }]);

    try {
      const res = await fetch("/api/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, thread_id: threadId.current }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const reply: string = data.reply ?? "";

      let audioUrl: string | undefined;
      try {
        const ttsRes = await fetch("/api/voice/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: reply }),
        });
        if (ttsRes.ok) {
          const blob = await ttsRes.blob();
          audioUrl = URL.createObjectURL(blob);
        }
      } catch {
        // TTS is best-effort; missing API key is fine in dev
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: reply, audioUrl, sources: data.sources ?? [] },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-full max-w-2xl mx-auto">
      <div className="flex-1 overflow-y-auto space-y-4 p-4">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-prose rounded-2xl px-4 py-3 text-sm ${
                m.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap">{m.text}</p>
              {m.audioUrl && (
                <audio
                  src={m.audioUrl}
                  controls
                  autoPlay
                  className="mt-2 w-full"
                />
              )}
              {m.sources && <SourcePlayer sources={m.sources} />}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-2xl px-4 py-3 text-gray-400 text-sm animate-pulse">
              Thinking…
            </div>
          </div>
        )}
      </div>

      <div className="border-t border-gray-800 p-4 space-y-3">
        <AudioRecorder onTranscript={(t) => sendMessage(t)} />
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
            placeholder="Type a question…"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 text-sm focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-4 py-2 rounded-xl text-sm font-semibold"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
