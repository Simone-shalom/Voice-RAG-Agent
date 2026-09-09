import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { voiceStreamUrl, connectVoiceStream } from "../voiceWebSocket";

describe("voiceStreamUrl", () => {
  it("uses NEXT_PUBLIC_WS_URL directly when set", () => {
    vi.stubEnv("NEXT_PUBLIC_WS_URL", "ws://localhost:8000");
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://backend:8000"); // compose-internal, must NOT be used
    expect(voiceStreamUrl("t1")).toBe("ws://localhost:8000/voice/stream?thread_id=t1");
  });

  it("falls back to deriving from NEXT_PUBLIC_API_URL when NEXT_PUBLIC_WS_URL is unset (https -> wss)", () => {
    vi.stubEnv("NEXT_PUBLIC_WS_URL", "");
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://backend.example.com");
    expect(voiceStreamUrl("t1")).toBe("wss://backend.example.com/voice/stream?thread_id=t1");
  });

  it("falls back to deriving from NEXT_PUBLIC_API_URL when NEXT_PUBLIC_WS_URL is unset (http -> ws)", () => {
    vi.stubEnv("NEXT_PUBLIC_WS_URL", "");
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://localhost:8000");
    expect(voiceStreamUrl("t1")).toBe("ws://localhost:8000/voice/stream?thread_id=t1");
  });
});

describe("connectVoiceStream", () => {
  let sockets: any[];

  beforeEach(() => {
    sockets = [];
    const MockWebSocket: any = vi.fn().mockImplementation(function (this: any) {
      this.send = vi.fn();
      this.close = vi.fn();
      this.readyState = MockWebSocket.OPEN;
      sockets.push(this);
      return this;
    });
    MockWebSocket.CONNECTING = 0;
    MockWebSocket.OPEN = 1;
    MockWebSocket.CLOSING = 2;
    MockWebSocket.CLOSED = 3;
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  it("forwards parsed JSON messages to onMessage", () => {
    const onMessage = vi.fn();
    connectVoiceStream("t1", { onMessage });
    const socket = sockets[0];
    socket.onmessage({ data: JSON.stringify({ type: "token", text: "hi" }) });
    expect(onMessage).toHaveBeenCalledWith({ type: "token", text: "hi" });
  });

  it("send() forwards to the underlying socket when OPEN", () => {
    const controls = connectVoiceStream("t1", { onMessage: vi.fn() });
    controls.send("hello");
    expect(sockets[0].send).toHaveBeenCalledWith("hello");
  });

  it("send() drops the payload without throwing when the socket is not OPEN", () => {
    const controls = connectVoiceStream("t1", { onMessage: vi.fn() });
    sockets[0].readyState = (globalThis as any).WebSocket.CLOSED;
    expect(() => controls.send("hello")).not.toThrow();
    expect(sockets[0].send).not.toHaveBeenCalled();
  });
});

describe("connectVoiceStream reconnect backoff", () => {
  let sockets: any[];

  beforeEach(() => {
    vi.useFakeTimers();
    sockets = [];
    const MockWebSocket: any = vi.fn().mockImplementation(function (this: any) {
      this.send = vi.fn();
      this.close = vi.fn();
      this.readyState = MockWebSocket.OPEN;
      sockets.push(this);
      return this;
    });
    MockWebSocket.CONNECTING = 0;
    MockWebSocket.OPEN = 1;
    MockWebSocket.CLOSING = 2;
    MockWebSocket.CLOSED = 3;
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("close() during pending backoff prevents reconnect attempt", () => {
    const controls = connectVoiceStream("t1", { onMessage: vi.fn() });
    const initialSocket = sockets[0];

    // Simulate an unexpected close that schedules a reconnect
    initialSocket.onclose();
    expect(sockets.length).toBe(1); // Only initial socket so far

    // Call close() before the backoff timer fires
    controls.close();

    // Advance time past the first backoff delay (500ms)
    vi.advanceTimersByTime(500);

    // No second socket should have been created
    expect(sockets.length).toBe(1);
  });

  it("stops reconnecting after 3 failed attempts", () => {
    const controls = connectVoiceStream("t1", { onMessage: vi.fn() });

    // First attempt: unexpected close, schedules reconnect with 500ms delay
    sockets[0].onclose();
    vi.advanceTimersByTime(500);
    expect(sockets.length).toBe(2); // New socket created

    // Second attempt: unexpected close, schedules reconnect with 1500ms delay
    sockets[1].onclose();
    vi.advanceTimersByTime(1500);
    expect(sockets.length).toBe(3); // Another new socket

    // Third attempt: unexpected close, schedules reconnect with 4000ms delay
    sockets[2].onclose();
    vi.advanceTimersByTime(4000);
    expect(sockets.length).toBe(4); // Another new socket

    // Fourth attempt: unexpected close, but we've exhausted our 3 attempts
    sockets[3].onclose();

    // Advance time by a large amount — no new socket should be created
    vi.advanceTimersByTime(10000);
    expect(sockets.length).toBe(4); // No additional socket
  });

  it("calls onGaveUp exactly once when the reconnect budget is exhausted", () => {
    const onGaveUp = vi.fn();
    connectVoiceStream("t1", { onMessage: vi.fn(), onGaveUp });

    sockets[0].onclose();
    vi.advanceTimersByTime(500);
    sockets[1].onclose();
    vi.advanceTimersByTime(1500);
    sockets[2].onclose();
    vi.advanceTimersByTime(4000);
    expect(onGaveUp).not.toHaveBeenCalled();

    // Fourth close: reconnect budget (3 attempts) is exhausted — give up.
    sockets[3].onclose();
    expect(onGaveUp).toHaveBeenCalledTimes(1);

    // Further time passing must not call it again or reconnect further.
    vi.advanceTimersByTime(10000);
    expect(onGaveUp).toHaveBeenCalledTimes(1);
    expect(sockets.length).toBe(4);
  });

  it("does not call onGaveUp when close() was deliberate", () => {
    const onGaveUp = vi.fn();
    const controls = connectVoiceStream("t1", { onMessage: vi.fn(), onGaveUp });
    controls.close();
    sockets[0].onclose();
    expect(onGaveUp).not.toHaveBeenCalled();
  });

  it("resets attempt counter after successful reconnect", () => {
    const controls = connectVoiceStream("t1", { onMessage: vi.fn() });

    // First sequence: close, reconnect with 500ms delay
    sockets[0].onclose();
    vi.advanceTimersByTime(500);
    expect(sockets.length).toBe(2);

    // Simulate successful open on the second socket
    sockets[1].onopen();

    // Now close unexpectedly again — should use the FIRST backoff delay (500ms), not 1500ms
    sockets[1].onclose();

    // Advance 500ms and verify a new socket is created (proving 500ms was used, not 1500ms)
    vi.advanceTimersByTime(500);
    expect(sockets.length).toBe(3);
  });

  it("defense in depth: open() checks deliberatelyClosed at entry", () => {
    const controls = connectVoiceStream("t1", { onMessage: vi.fn() });

    // This is a defensive test: if somehow a timer fires after close() is called,
    // open() should bail out immediately. We can't easily simulate this in the test,
    // but we can at least verify the close() behavior works correctly.
    sockets[0].onclose();
    controls.close();

    // Even after advancing timers, no reconnect should occur
    vi.advanceTimersByTime(10000);
    expect(sockets.length).toBe(1);
  });
});
