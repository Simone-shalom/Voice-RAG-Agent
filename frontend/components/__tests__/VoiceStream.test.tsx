import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, expect, it, beforeEach } from "vitest";
import VoiceStream from "../VoiceStream";
import type { Message } from "../ChatUI";

vi.mock("../../lib/voiceWebSocket", () => ({
  connectVoiceStream: vi.fn(),
}));

// Replace the real AudioRecorder with a minimal stand-in that keeps the
// same accessible name (so the "falls back to AudioRecorder UI" test still
// matches real markup) but is otherwise decoupled from its internal
// recording/fetch flow — this file only needs to prove VoiceStream wires
// its onTranscript prop through correctly, not re-test AudioRecorder's own
// (separately covered) recording behavior.
vi.mock("../AudioRecorder", () => ({
  default: ({ onTranscript }: { onTranscript: (text: string) => void }) => (
    <button aria-label="Start voice recording" onClick={() => onTranscript("mock transcript")}>
      Ask with Voice
    </button>
  ),
}));

import { connectVoiceStream } from "../../lib/voiceWebSocket";

describe("VoiceStream", () => {
  let onMessageHandler: (event: Record<string, unknown>) => void;

  beforeEach(() => {
    (connectVoiceStream as any).mockImplementation((_threadId: string, handlers: any) => {
      onMessageHandler = handlers.onMessage;
      handlers.onOpen?.();
      return { send: vi.fn(), close: vi.fn() };
    });
    (globalThis.navigator.mediaDevices as any).getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    // Fresh per test — the vitest.setup.ts FakeAudioContext stashes the
    // most recently created analyser here so a test can reach it (there's
    // no other way in from outside the component's effect).
    (globalThis as any).__lastFakeAnalyser = undefined;
    // Fresh per test — see FakeMediaRecorder in vitest.setup.ts.
    (globalThis as any).__fakeMediaRecorders = [];
  });

  it("shows a latency badge on a latency event", async () => {
    render(<VoiceStream setMessages={vi.fn()} threadId="t1" />);
    onMessageHandler({ type: "latency", ms: 642 });
    await waitFor(() => expect(screen.getByText(/642/)).toBeInTheDocument());
  });

  it("falls back to AudioRecorder UI on a fallback event", async () => {
    render(<VoiceStream setMessages={vi.fn()} threadId="t1" />);
    onMessageHandler({ type: "fallback", reason: "Deepgram unavailable" });
    // AudioRecorder's button carries aria-label="Start voice recording",
    // which wins over its visible "Ask with Voice" text for the
    // accessible name — query by the actual accessible name rather than
    // the visible label the brief's draft test used (that query can
    // never match against the real component).
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /start voice recording/i })).toBeInTheDocument()
    );
  });

  it("wires the fallback AudioRecorder's onTranscript through to the caller (Finding 1)", async () => {
    const onTranscript = vi.fn();
    render(<VoiceStream setMessages={vi.fn()} threadId="t1" onTranscript={onTranscript} />);
    onMessageHandler({ type: "fallback", reason: "Deepgram unavailable" });

    const button = await screen.findByRole("button", { name: /start voice recording/i });
    await userEvent.click(button);

    expect(onTranscript).toHaveBeenCalledWith("mock transcript");
  });

  it("creates a placeholder message pair on a voice turn's first token, without corrupting an existing assistant message (Finding 2)", async () => {
    let messages: Message[] = [
      { role: "user", text: "typed earlier" },
      { role: "assistant", text: "old typed answer", streaming: false },
    ];
    const setMessages = vi.fn((updater: any) => {
      messages = typeof updater === "function" ? updater(messages) : updater;
    });

    render(<VoiceStream setMessages={setMessages as any} threadId="t1" />);
    await waitFor(() => expect((globalThis as any).__lastFakeAnalyser).toBeTruthy());

    onMessageHandler({ type: "token", text: "Hi" });

    // The old assistant message must be untouched...
    expect(messages[1]).toEqual({ role: "assistant", text: "old typed answer", streaming: false });
    // ...and a fresh user+assistant pair appended, with the token applied
    // to the NEW assistant placeholder, not dropped or merged into the old one.
    expect(messages).toHaveLength(4);
    expect(messages[2]).toMatchObject({ role: "user", text: expect.any(String) });
    expect(messages[3]).toMatchObject({ role: "assistant", text: "Hi", streaming: true });
  });

  it("detects speech during an answer and aborts playback + sends a cancel frame exactly once", async () => {
    const sendSpy = vi.fn();
    (connectVoiceStream as any).mockImplementation((_threadId: string, handlers: any) => {
      onMessageHandler = handlers.onMessage;
      handlers.onOpen?.();
      return { send: sendSpy, close: vi.fn() };
    });

    render(<VoiceStream setMessages={vi.fn()} threadId="t1" />);

    // Wait for the mic pipeline (barge-in VAD analyser) to be wired up
    // after getUserMedia resolves.
    await waitFor(() => expect((globalThis as any).__lastFakeAnalyser).toBeTruthy());

    // Start a turn — the detection loop only acts while answeringRef is true.
    onMessageHandler({ type: "token", text: "Hello" });

    // Simulate the user speaking: loud samples on the next animation frame.
    // Held steady across frames — the detector now requires a few
    // consecutive above-threshold frames (debounce), not just one.
    (globalThis as any).__lastFakeAnalyser._amplitude = 100;

    await waitFor(() =>
      expect(sendSpy).toHaveBeenCalledWith(JSON.stringify({ type: "cancel" }))
    );

    // Keep "talking" for a few more frames — must not flood cancel frames.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sendSpy).toHaveBeenCalledTimes(1);
  });

  it("falls back to AudioRecorder when the connection never opens (reconnect budget exhausted)", async () => {
    (connectVoiceStream as any).mockImplementation((_threadId: string, handlers: any) => {
      onMessageHandler = handlers.onMessage;
      // Deliberately never call handlers.onOpen() — simulates a
      // connection that never opens at all (backend down, Origin
      // rejected, etc.) so the only signal VoiceStream gets is onGaveUp.
      return {
        send: vi.fn(),
        close: vi.fn(),
        __handlers: handlers,
      };
    });

    render(<VoiceStream setMessages={vi.fn()} threadId="t1" />);

    // Grab the handlers this render's connectVoiceStream call was given
    // and fire the "reconnect budget exhausted" signal directly.
    const call = (connectVoiceStream as any).mock.calls[(connectVoiceStream as any).mock.calls.length - 1];
    const handlers = call[1];
    handlers.onGaveUp();

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /start voice recording/i })).toBeInTheDocument()
    );
  });

  it("restarts MediaRecorder on every reconnect after the first open (Fix 3b)", async () => {
    let handlers: any;
    (connectVoiceStream as any).mockImplementation((_threadId: string, h: any) => {
      handlers = h;
      onMessageHandler = h.onMessage;
      h.onOpen?.();
      return { send: vi.fn(), close: vi.fn() };
    });

    render(<VoiceStream setMessages={vi.fn()} threadId="t1" />);

    await waitFor(() => expect((globalThis as any).__fakeMediaRecorders.length).toBe(1));
    const first = (globalThis as any).__fakeMediaRecorders[0];
    expect(first.state).toBe("recording");

    // Simulate a reconnect: a new backend connection opens (backend closes
    // the WS after every turn by design).
    handlers.onOpen();

    await waitFor(() => expect((globalThis as any).__fakeMediaRecorders.length).toBe(2));
    const second = (globalThis as any).__fakeMediaRecorders[1];
    expect(first.state).toBe("inactive"); // old recorder was stopped
    expect(first.ondataavailable).toBeNull(); // detached before stop, so its final flush isn't forwarded
    expect(second.state).toBe("recording"); // fresh recorder started for the new connection
  });

  it("seeds a placeholder assistant message when an error arrives as the first event of a turn (Fix 6)", async () => {
    let messages: Message[] = [];
    const setMessages = vi.fn((updater: any) => {
      messages = typeof updater === "function" ? updater(messages) : updater;
    });

    render(<VoiceStream setMessages={setMessages as any} threadId="t1" />);
    await waitFor(() => expect((globalThis as any).__lastFakeAnalyser).toBeTruthy());

    // No prior token/audio event — error is the very first event of this turn.
    onMessageHandler({ type: "error", text: "ANTHROPIC_API_KEY not set" });

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({ role: "user" });
    expect(messages[1]).toMatchObject({
      role: "assistant",
      text: "Error: ANTHROPIC_API_KEY not set",
      streaming: false,
    });
  });

  it("does not resurrect a stale effect instance when threadId changes while getUserMedia is pending (Fix 7)", async () => {
    const resolvers: Array<(stream: any) => void> = [];
    let callIndex = 0;
    (globalThis.navigator.mediaDevices as any).getUserMedia = vi.fn().mockImplementation(() => {
      const idx = callIndex++;
      return new Promise((resolve) => {
        resolvers[idx] = resolve;
      });
    });

    (connectVoiceStream as any).mockImplementation((_threadId: string, handlers: any) => {
      onMessageHandler = handlers.onMessage;
      handlers.onOpen?.();
      return { send: vi.fn(), close: vi.fn() };
    });

    const setMessages = vi.fn();
    const { rerender } = render(<VoiceStream setMessages={setMessages} threadId="t1" />);

    await waitFor(() => expect(resolvers.length).toBe(1));

    // threadId changes before the first getUserMedia call resolves —
    // React runs the first effect instance's cleanup (which, pre-fix, set
    // a SHARED unmountedRef true) and immediately starts a second
    // instance's effect body (which, pre-fix, reset that same shared ref
    // back to false) in the same commit.
    rerender(<VoiceStream setMessages={setMessages} threadId="t2" />);

    await waitFor(() => expect(resolvers.length).toBe(2));

    // Now let the STALE (first) getUserMedia call resolve, well after its
    // owning effect instance's cleanup already ran. Pre-fix, the shared
    // ref would read `false` here (reset by the second instance) and this
    // stale .then() would wrongly proceed to start a MediaRecorder/rAF
    // loop nothing will ever tear down.
    const stopFirst = vi.fn();
    resolvers[0]({ getTracks: () => [{ stop: stopFirst }] });

    await waitFor(() => expect(stopFirst).toHaveBeenCalled());

    // The stale instance must NOT have started a MediaRecorder — only the
    // live (second) instance should, once its own getUserMedia resolves.
    expect((globalThis as any).__fakeMediaRecorders.length).toBe(0);

    const stopSecond = vi.fn();
    resolvers[1]({ getTracks: () => [{ stop: stopSecond }] });

    await waitFor(() => expect((globalThis as any).__fakeMediaRecorders.length).toBe(1));
    expect(stopSecond).not.toHaveBeenCalled(); // the live stream stays open
  });

  it("restarts MediaRecorder on a reconnect that arrives AFTER getUserMedia already started it (turn-1 header race, residual Fix A)", async () => {
    // Reproduces the ordering the old `hasOpenedOnce` flag got wrong:
    // getUserMedia resolves (and starts the first recorder) BEFORE the WS
    // handshake completes. The old code would treat this first onOpen as
    // "the first open" and just set a flag and return, leaving the
    // already-running recorder's header-bearing first blob to have been
    // silently dropped by the readyState guard while the socket was still
    // CONNECTING — breaking turn 1, not just reconnects. The fix restarts
    // the recorder whenever one already exists at onOpen time, regardless
    // of whether this is the first open or a later one.
    let handlers: any;
    (connectVoiceStream as any).mockImplementation((_threadId: string, h: any) => {
      handlers = h;
      onMessageHandler = h.onMessage;
      // Deliberately do NOT call onOpen synchronously here — simulates a
      // WS handshake that takes longer than getUserMedia, so a recorder
      // already exists by the time onOpen eventually fires.
      return { send: vi.fn(), close: vi.fn() };
    });

    render(<VoiceStream setMessages={vi.fn()} threadId="t1" />);

    // getUserMedia resolves and starts the first recorder before the WS
    // has opened at all.
    await waitFor(() => expect((globalThis as any).__fakeMediaRecorders.length).toBe(1));
    const first = (globalThis as any).__fakeMediaRecorders[0];
    expect(first.state).toBe("recording");

    // The WS finally opens — this is the FIRST onOpen call for this
    // connection, but a recorder already exists.
    handlers.onOpen();

    await waitFor(() => expect((globalThis as any).__fakeMediaRecorders.length).toBe(2));
    const second = (globalThis as any).__fakeMediaRecorders[1];
    expect(first.state).toBe("inactive"); // old recorder was stopped
    expect(first.ondataavailable).toBeNull(); // detached before stop
    expect(second.state).toBe("recording"); // fresh recorder, header sent over the now-open socket
  });

  it("releases the answering lock on a reconnect that arrives mid-answer (abnormal drop, residual Fix C)", async () => {
    // If the WS drops mid-answer without a terminal event (done/
    // cancelled/error) and then reconnects, the shared "answering" flag
    // must not stay stuck forever — that would also permanently disable
    // ChatUI's typed input with no recovery but a page reload. A
    // reconnect (recorder already exists) happening while still
    // "answering" is always an abnormal-drop signal, so it must release
    // the lock.
    let handlers: any;
    const onAnsweringChange = vi.fn();
    (connectVoiceStream as any).mockImplementation((_threadId: string, h: any) => {
      handlers = h;
      onMessageHandler = h.onMessage;
      h.onOpen?.(); // initial open
      return { send: vi.fn(), close: vi.fn() };
    });

    render(
      <VoiceStream setMessages={vi.fn()} threadId="t1" onAnsweringChange={onAnsweringChange} />
    );
    await waitFor(() => expect((globalThis as any).__lastFakeAnalyser).toBeTruthy());

    // Start a turn but never deliver a terminal event — simulates the
    // socket dropping mid-answer.
    onMessageHandler({ type: "token", text: "Hi" });
    expect(onAnsweringChange).toHaveBeenLastCalledWith(true);

    // The connection drops and a fresh one reconnects (same path Fix A
    // exercises) while the previous turn never reached "done".
    handlers.onOpen();

    expect(onAnsweringChange).toHaveBeenLastCalledWith(false);
  });

  it("reports voice-turn streaming transitions via onAnsweringChange (Fix 5)", async () => {
    const onAnsweringChange = vi.fn();
    render(<VoiceStream setMessages={vi.fn()} threadId="t1" onAnsweringChange={onAnsweringChange} />);
    await waitFor(() => expect((globalThis as any).__lastFakeAnalyser).toBeTruthy());

    onMessageHandler({ type: "token", text: "Hi" });
    expect(onAnsweringChange).toHaveBeenLastCalledWith(true);

    onMessageHandler({ type: "done" });
    expect(onAnsweringChange).toHaveBeenLastCalledWith(false);
  });
});
