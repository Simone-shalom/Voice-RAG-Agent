"use client";

import { useEffect, useRef, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import { connectVoiceStream } from "@/lib/voiceWebSocket";
import { applyStreamEvent, type Message } from "./ChatUI";

interface Props {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  threadId: string;
  onTranscript?: (text: string) => void;
  // Reports every transition of this component's own "a voice turn is
  // actively streaming" flag up to the caller (ChatUI), so it can be
  // combined with ChatUI's own `loading` state into one shared guard that
  // stops a typed send and a voice turn from mutating the same message
  // bubble concurrently. See ChatUI.tsx's `voiceAnswering` state.
  onAnsweringChange?: (answering: boolean) => void;
}

// Same RMS amplitude threshold AudioRecorder.tsx uses for silence
// detection (auto-stop once the user has gone quiet). Reused here for the
// mirror-image purpose: detecting that the user has started speaking again
// while the assistant is answering (barge-in).
const SPEECH_ONSET_THRESHOLD = 0.015;
// Require a few consecutive above-threshold analysis frames before acting
// (~50ms at typical rAF cadence) rather than a single ~46ms window, so a
// keyboard click, cough, or chair creak while the assistant is answering
// (a text input sits directly under this component — typing while
// listening to an answer is a completely normal interaction) doesn't
// false-trigger an interruption. Negligible cost against a Deepgram
// round-trip.
const MIN_CONSECUTIVE_SPEECH_FRAMES = 3;

export default function VoiceStream({ setMessages, threadId, onTranscript, onAnsweringChange }: Props) {
  const [fallback, setFallback] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const controlsRef = useRef<{ send: (d: Blob | string) => void; close: () => void } | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  // Chained playback queue (same pattern as ChatUI's enqueueAudioChunk) —
  // a turn's audio events arrive as several sentence-sized chunks, and
  // without sequencing them, each decodeAudioData().then(...) races the
  // others and multiple sources can start and play concurrently. Only
  // the *last-started* source would then be reachable via
  // currentSourceRef, so abortPlayback() would silence just that one and
  // leave earlier overlapping chunks still audible.
  const audioQueueRef = useRef<Promise<void>>(Promise.resolve());
  // Bumped on every abort; a chunk queued before the abort checks this
  // before it actually plays and skips itself if it's stale, so a
  // barge-in also discards not-yet-started queued chunks, not just the
  // currently-playing one.
  const abortTokenRef = useRef(0);
  const answeringRef = useRef(false);
  // Guards against re-sending {"type":"cancel"} on every animation frame
  // while the user keeps talking after the first onset was detected.
  // Reset whenever a new turn starts.
  const cancelSentRef = useRef(false);
  // Consecutive above-threshold frame count for the debounce above; reset
  // on any below-threshold frame and whenever we're not mid-answer.
  const consecutiveSpeechFramesRef = useRef(0);
  // Whether this connection has ever successfully opened at least once —
  // informational (surfaces "never connected at all" vs. "connected, then
  // lost it" if this ever needs a distinct message), not required for the
  // fallback decision itself: any exhausted-reconnect signal falls back,
  // regardless of prior successful opens (see onGaveUp below).
  const everOpenedRef = useRef(false);
  // Latest onAnsweringChange, read (not closed-over) inside the effect —
  // ChatUI typically passes an inline arrow function whose identity
  // changes on every ChatUI render, and this effect's deps deliberately
  // don't include it (adding it would tear down and reopen the voice
  // WebSocket on every unrelated ChatUI re-render, e.g. every typed
  // token).
  const onAnsweringChangeRef = useRef(onAnsweringChange);
  onAnsweringChangeRef.current = onAnsweringChange;

  // Barge-in voice-activity detection: a SECOND, separate AudioContext
  // pipeline over the raw mic stream — distinct from audioCtxRef above,
  // which is for TTS *playback*, not mic input. Same input-analysis vs.
  // playback split AudioRecorder.tsx already uses for its own silence
  // detection (see its `audioCtxRef`/`analyserRef` there).
  const vadCtxRef = useRef<AudioContext | null>(null);
  const vadAnalyserRef = useRef<AnalyserNode | null>(null);
  const vadRafRef = useRef<number | null>(null);

  function setAnswering(value: boolean) {
    answeringRef.current = value;
    onAnsweringChangeRef.current?.(value);
  }

  function abortPlayback() {
    abortTokenRef.current += 1;
    currentSourceRef.current?.stop();
    currentSourceRef.current = null;
    audioQueueRef.current = Promise.resolve();
  }

  function playAudioChunk(base64: string) {
    const myToken = abortTokenRef.current;
    audioQueueRef.current = audioQueueRef.current.then(async () => {
      if (myToken !== abortTokenRef.current) return; // aborted before its turn came up
      const ctx = audioCtxRef.current ?? new AudioContext();
      audioCtxRef.current = ctx;
      if (ctx.state === "suspended") await ctx.resume();
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      try {
        const buffer = await ctx.decodeAudioData(bytes.buffer);
        if (myToken !== abortTokenRef.current) return; // aborted while decoding
        await new Promise<void>((resolve) => {
          const src = ctx.createBufferSource();
          src.buffer = buffer;
          src.connect(ctx.destination);
          src.onended = () => resolve();
          currentSourceRef.current = src;
          src.start();
        });
      } catch {
        // decode/playback failure is non-fatal
      }
    });
  }

  // Runs every animation frame for the life of the component. Only
  // *acts* while a turn might be streaming/playing (answeringRef) — no
  // point burning cycles or risking false cancels during the normal
  // "user is asking a question" phase, which the continuous MediaRecorder
  // capture already handles for STT.
  function monitorForBargeIn() {
    const analyser = vadAnalyserRef.current;
    if (analyser && answeringRef.current && !cancelSentRef.current) {
      const data = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(data);
      let sumSquares = 0;
      for (let i = 0; i < data.length; i++) {
        const normalized = (data[i] - 128) / 128;
        sumSquares += normalized * normalized;
      }
      const rms = Math.sqrt(sumSquares / data.length);
      if (rms > SPEECH_ONSET_THRESHOLD) {
        consecutiveSpeechFramesRef.current += 1;
        if (consecutiveSpeechFramesRef.current >= MIN_CONSECUTIVE_SPEECH_FRAMES) {
          // The user started talking again — cut local playback
          // immediately (client-side, instant) and tell the server too,
          // so it can cancel any in-flight generation/TTS. The backend's
          // socket reader (backend/app/voice/stream_ws.py) stays alive
          // for the whole turn specifically so this frame can reach it
          // while stream_agent_response is still streaming.
          cancelSentRef.current = true;
          abortPlayback();
          controlsRef.current?.send(JSON.stringify({ type: "cancel" }));
        }
      } else {
        consecutiveSpeechFramesRef.current = 0;
      }
    } else {
      consecutiveSpeechFramesRef.current = 0;
    }
    vadRafRef.current = requestAnimationFrame(monitorForBargeIn);
  }

  useEffect(() => {
    if (fallback) return;
    // Per-effect-instance guard, closed over by both the getUserMedia
    // .then() callback and this effect's cleanup — NOT a ref. A ref
    // shared across effect re-runs (e.g. on a threadId change) gets reset
    // to `false` by the new effect body in the same commit that the
    // previous instance's cleanup just set it `true` in, so a
    // still-pending getUserMedia() promise from the OLD instance would
    // see the fresh `false` and proceed as if it still owned this mount —
    // leaking a mic stream and orphaning a rAF loop. A local `let`
    // captured per-instance can't be resurrected by a later effect run.
    let cancelled = false;

    function startRecorder(s: MediaStream): MediaRecorder {
      const r = new MediaRecorder(s);
      r.ondataavailable = (e) => controlsRef.current?.send(e.data);
      r.start(250);
      return r;
    }

    // Declared BEFORE connectVoiceStream() below, not after — onOpen (used
    // by connectVoiceStream to fire the reconnect logic) can now run on
    // the FIRST open too (see Fix A above), and connectVoiceStream may in
    // principle invoke it synchronously/re-entrantly during its own call.
    // If these were declared after that call (as `let`, so still in the
    // temporal dead zone), such a synchronous onOpen would throw
    // "Cannot access 'stream'/'recorder' before initialization" instead of
    // seeing them as not-yet-assigned `null`.
    let recorder: MediaRecorder | null = null;
    let stream: MediaStream | null = null;

    const controls = connectVoiceStream(threadId, {
      onOpen: () => {
        everOpenedRef.current = true;
        // The backend closes its WebSocket after every single voice turn
        // by design (one connection per turn, fresh Deepgram session
        // each time). MediaRecorder only emits a WebM/EBML container
        // header in the very first blob after start() — every blob
        // after that is a headerless media cluster the NEW backend
        // connection's Deepgram session can't parse. So every
        // reconnection needs a fresh MediaRecorder instance (same
        // underlying MediaStream — no need to re-prompt for mic access)
        // to guarantee the next backend connection gets a valid header.
        //
        // Restart unconditionally whenever a recorder ALREADY EXISTS at
        // the moment this fires — that's the reconnect signal, not "is
        // this the first open". This correctly handles both orderings of
        // the WS-open vs. getUserMedia race:
        //  - WS opens before getUserMedia resolves: no recorder exists
        //    yet, so nothing happens here — the getUserMedia().then()
        //    callback below creates the first recorder afterward,
        //    against an already-open socket, so its header-bearing first
        //    blob is sent while readyState is OPEN.
        //  - getUserMedia resolves before the WS opens: a recorder is
        //    already running and may have already emitted its header
        //    blob into the socket while it was still CONNECTING (dropped
        //    by the readyState guard in lib/voiceWebSocket). Stopping and
        //    restarting it here, now that the socket is actually open,
        //    discards that stale state and sends a fresh header over the
        //    now-open connection. (Previously this only fired on every
        //    open AFTER the first, via a `hasOpenedOnce` flag — which
        //    missed exactly this ordering and could silently break even
        //    turn 1.)
        if (stream && recorder) {
          // Detach the old recorder's handler before stopping it —
          // MediaRecorder.stop() fires one final "dataavailable" with
          // whatever was buffered, which would otherwise forward a
          // stale, headerless chunk from the OLD muxer onto the BRAND
          // NEW connection right as it opens — reintroducing the exact
          // bug this restart exists to fix.
          recorder.ondataavailable = null;
          recorder.stop();
          recorder = startRecorder(stream);
          // Fix C: a reconnect happening while `answeringRef` is still
          // true means the previous turn's terminal event (done/
          // cancelled/error) never arrived — the connection dropped
          // mid-answer instead. The backend closes the socket after
          // every completed turn by design, so reaching a reconnect here
          // with the lock still held is always an abnormal-drop signal,
          // never a legitimate in-progress turn. Release it now, or the
          // shared "answering" flag (which also disables ChatUI's typed
          // input) would stay stuck forever with no recovery but a page
          // reload.
          if (answeringRef.current) {
            setAnswering(false);
          }
        }
      },
      onGaveUp: () => {
        // Reconnect budget exhausted without a working connection —
        // there's no path back to streaming voice for this mount. Fall
        // back to the existing push-to-talk AudioRecorder flow rather
        // than leaving a hot mic and a UI stuck on "Listening" forever.
        setFallback(true);
      },
      onMessage: (event) => {
        if (event.type === "fallback") {
          setFallback(true);
          return;
        }
        if (event.type === "latency" && typeof event.ms === "number") {
          setLatencyMs(event.ms);
        }
        const startsNewTurn =
          (event.type === "token" || event.type === "audio" || event.type === "error") &&
          !answeringRef.current;
        if (startsNewTurn) {
          // first event of a new turn — a prior turn's playback (if any)
          // must stop immediately, this is the barge-in path
          abortPlayback();
          setAnswering(true);
          cancelSentRef.current = false;
          // Seed a fresh message pair the same way ChatUI's typed-path
          // sendMessage() does before its own SSE loop starts. Without
          // this, applyStreamEvent's token/audio/error handling — which
          // only ever mutates an EXISTING last-assistant message — either
          // silently drops every voice-turn token (fresh thread, no
          // assistant message yet) or appends onto/overwrites a stale
          // assistant message left over from an earlier typed turn
          // (visible corruption). This also covers an `error` event
          // arriving as the very FIRST event of a turn (e.g. a missing
          // ANTHROPIC_API_KEY) — without a placeholder to attach to, that
          // error was previously either dropped entirely or landed on an
          // unrelated stale message. The backend never sends the
          // transcript back, so the user bubble is a placeholder rather
          // than the real text.
          setMessages((prev) => [
            ...prev,
            { role: "user", text: "🎤 (voice message)" },
            { role: "assistant", text: "", streaming: true },
          ]);
        }
        const isTerminal = applyStreamEvent(
          event as { type: string; text?: string; data?: string | never[] },
          setMessages,
          playAudioChunk
        );
        if (isTerminal) {
          setAnswering(false);
          cancelSentRef.current = false;
        }
      },
    });
    controlsRef.current = controls;

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((s) => {
        if (cancelled) {
          // Component unmounted (or this effect instance was superseded
          // by a re-run) while the permission prompt/getUserMedia call
          // was pending — this instance's cleanup already ran, so
          // release this stream immediately instead of starting a
          // MediaRecorder, AudioContext, and rAF loop that nothing will
          // ever tear down.
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        recorder = startRecorder(s);

        const vadCtx = new AudioContext();
        const source = vadCtx.createMediaStreamSource(s);
        const analyser = vadCtx.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        if (vadCtx.state === "suspended") {
          // Browsers (Chrome in particular) create a new AudioContext
          // suspended unless it's constructed inside a user-gesture call
          // stack — this one is created inside a getUserMedia().then()
          // callback, not a click handler, so it's exposed to that. A
          // suspended context doesn't pull data through its
          // MediaStreamSourceNode, so getByteTimeDomainData() would
          // silently return a flat buffer and barge-in would never fire
          // with no error anywhere.
          vadCtx.resume().catch(() => {});
        }
        vadCtxRef.current = vadCtx;
        vadAnalyserRef.current = analyser;
        vadRafRef.current = requestAnimationFrame(monitorForBargeIn);
      })
      .catch(() => setFallback(true));

    return () => {
      cancelled = true;
      controls.close();
      recorder?.stop();
      stream?.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close().catch(() => {});
      if (vadRafRef.current !== null) {
        cancelAnimationFrame(vadRafRef.current);
        vadRafRef.current = null;
      }
      vadCtxRef.current?.close().catch(() => {});
      vadCtxRef.current = null;
      vadAnalyserRef.current = null;
      // Release the shared "a turn is streaming" lock if this instance
      // unmounts (e.g. threadId changes) mid-turn — otherwise ChatUI's
      // typed-send guard would stay stuck disabled forever.
      if (answeringRef.current) {
        setAnswering(false);
      }
    };
  }, [threadId, fallback, setMessages]);

  if (fallback) {
    return <AudioRecorder onTranscript={onTranscript ?? (() => {})} />;
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <p className="text-xs text-gray-500">🎙️ Listening — talk any time, even mid-answer</p>
      {latencyMs !== null && (
        <p className="text-[10px] text-gray-600">⚡ {Math.round(latencyMs)}ms</p>
      )}
    </div>
  );
}
