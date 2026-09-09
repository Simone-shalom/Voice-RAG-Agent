import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement the Web Audio API — ChatUI creates an
// AudioContext synchronously on every send (to stay inside the
// user-gesture call stack for TTS playback), so it must exist even for
// tests that never touch audio output.
//
// VoiceStream's barge-in speech-detection loop also uses a *second*
// AudioContext (createMediaStreamSource + createAnalyser over the raw mic
// stream, mirroring AudioRecorder.tsx's own VAD pipeline) — FakeAnalyserNode
// gives tests a hook (`_amplitude`) to simulate "the user is talking" on
// the next getByteTimeDomainData() read, and createAnalyser() stashes the
// instance on globalThis.__lastFakeAnalyser so a test can reach the
// analyser created inside VoiceStream's effect (there is no other way in
// from outside the component).
class FakeAnalyserNode {
  fftSize = 2048;
  // Test hook — 0 (default) reads as silence; set a positive value to
  // simulate loud mic input on the next getByteTimeDomainData() call.
  _amplitude = 0;
  getByteTimeDomainData(arr: Uint8Array) {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = 128 + (i % 2 === 0 ? this._amplitude : -this._amplitude);
    }
  }
}

class FakeAudioContext {
  state: "running" | "suspended" | "closed" = "running";
  destination = {};
  resume() {
    this.state = "running";
    return Promise.resolve();
  }
  close() {
    this.state = "closed";
    return Promise.resolve();
  }
  decodeAudioData() {
    return Promise.resolve({});
  }
  createBufferSource() {
    return {
      buffer: null,
      connect() {},
      start() {},
      stop() {},
      onended: null as (() => void) | null,
    };
  }
  createAnalyser() {
    const analyser = new FakeAnalyserNode();
    // @ts-expect-error — test-only hook, see class comment above
    globalThis.__lastFakeAnalyser = analyser;
    return analyser;
  }
  createMediaStreamSource(_stream: MediaStream) {
    return { connect() {} };
  }
}
// @ts-expect-error — partial stub, sufficient for tests that don't assert on audio playback
globalThis.AudioContext = FakeAudioContext;

// jsdom has no MediaRecorder — VoiceStream creates one on mount to stream
// mic audio over the WS connection, so it must exist even for tests that
// never assert on actual encoded audio.
class FakeMediaRecorder {
  static isTypeSupported() {
    return true;
  }
  state: "inactive" | "recording" = "inactive";
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  constructor(_stream: MediaStream) {
    // Test hook (mirrors __lastFakeAnalyser above) — every instance ever
    // constructed is pushed here so a test can assert "a fresh
    // MediaRecorder was created on reconnect" (VoiceStream.tsx restarts
    // the recorder on every WS reconnect after the first, since each new
    // backend connection needs a fresh WebM container header). Reset it
    // per-test via `(globalThis as any).__fakeMediaRecorders = []`.
    ((globalThis as any).__fakeMediaRecorders ??= []).push(this);
  }
  start(_timeslice?: number) {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    this.onstop?.();
  }
}
// @ts-expect-error — partial stub, sufficient for tests that don't assert on actual encoded audio
globalThis.MediaRecorder = FakeMediaRecorder;

// jsdom has no getUserMedia. VoiceStream calls it unconditionally on mount
// (unlike AudioRecorder, which only calls it on button click), so every
// test that renders ChatUI/VoiceStream needs at least a safe default here
// — a bare `{}` would make `navigator.mediaDevices.getUserMedia(...)` throw
// "not a function" synchronously inside a useEffect instead of rejecting,
// which VoiceStream's .catch() can't handle. Reject by default so mount
// falls back to AudioRecorder gracefully; tests that care about the
// success path override this per-test via vi.stubGlobal or direct
// assignment (see VoiceStream.test.tsx).
// @ts-expect-error — partial stub, sufficient for tests that don't assert on real media capture
globalThis.navigator.mediaDevices = globalThis.navigator.mediaDevices ?? {
  getUserMedia: () => Promise.reject(new Error("getUserMedia not stubbed in this test")),
};
