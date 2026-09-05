"use client";

import { useRef, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import SourcePlayer from "./SourcePlayer";
import { Episode } from "./FileUploader";

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

interface Props {
  selectedEpisode: Episode | null;
}

export default function ChatUI({ selectedEpisode }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const threadId = useRef(crypto.randomUUID());

  async function sendMessage(userText: string) {
    if (!userText.trim()) return;
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text: userText }]);

    // prefix with episode context so the agent focuses on the selected file
    const messagePayload = selectedEpisode
      ? `[Focus on episode: "${selectedEpisode.filename}" (id: ${selectedEpisode.id})] ${userText}`
      : userText;

    try {
      const res = await fetch("/api/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messagePayload, thread_id: threadId.current }),
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
          audioUrl = URL.createObjectURL(await ttsRes.blob());
        }
      } catch {
        // TTS is best-effort; missing API key is acceptable in dev
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
      {/* episode focus chip */}
      {selectedEpisode && (
        <div className="shrink-0 px-4 pt-3">
          <div className="inline-flex items-center gap-1.5 bg-indigo-950/70 border border-indigo-700/60 rounded-full px-3 py-1 text-xs text-indigo-300">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
            <span className="truncate max-w-[240px]" title={selectedEpisode.filename}>
              Focused on: <span className="font-medium text-indigo-200">{selectedEpisode.filename}</span>
            </span>
            <span className="text-indigo-500 ml-0.5">·</span>
            <span className="text-indigo-500 text-[10px]">
              deselect in sidebar
            </span>
          </div>
        </div>
      )}

      {/* messages */}
      <div className="flex-1 overflow-y-auto space-y-4 p-4">
        {messages.length === 0 && !loading && (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-gray-600 text-center max-w-xs">
              {selectedEpisode
                ? `Focused on "${selectedEpisode.filename}". Ask anything about it.`
                : "Ask a question about your audio library, or select an episode in the sidebar to focus on it."}
            </p>
          </div>
        )}

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
                <audio src={m.audioUrl} controls autoPlay className="mt-2 w-full" />
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

      {/* input bar */}
      <div className="shrink-0 border-t border-gray-800 p-4 space-y-3">
        <AudioRecorder onTranscript={(t) => sendMessage(t)} />
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
            placeholder={
              selectedEpisode
                ? `Ask about "${selectedEpisode.filename}"…`
                : "Type a question…"
            }
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
