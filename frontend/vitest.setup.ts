import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement the Web Audio API — ChatUI creates an
// AudioContext synchronously on every send (to stay inside the
// user-gesture call stack for TTS playback), so it must exist even for
// tests that never touch audio output.
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
      onended: null as (() => void) | null,
    };
  }
}
// @ts-expect-error — partial stub, sufficient for tests that don't assert on audio playback
globalThis.AudioContext = FakeAudioContext;
