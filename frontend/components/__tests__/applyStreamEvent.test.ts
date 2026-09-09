import { describe, expect, it, vi } from "vitest";
import { applyStreamEvent, type Message } from "../ChatUI";

function seedMessages(): Message[] {
  return [
    { role: "user", text: "hi" },
    { role: "assistant", text: "", streaming: true },
  ];
}

describe("applyStreamEvent", () => {
  it("appends token text to the last assistant message", () => {
    let messages = seedMessages();
    const setMessages = (updater: any) => { messages = updater(messages); };
    applyStreamEvent({ type: "token", text: "Hello" }, setMessages, vi.fn());
    expect(messages[1].text).toBe("Hello");
  });

  it("calls onAudio for audio events", () => {
    const messages = seedMessages();
    const onAudio = vi.fn();
    applyStreamEvent({ type: "audio", data: "base64data" }, () => {}, onAudio);
    expect(onAudio).toHaveBeenCalledWith("base64data");
  });

  it("returns true and marks streaming false on done", () => {
    let messages = seedMessages();
    const setMessages = (updater: any) => { messages = updater(messages); };
    const isTerminal = applyStreamEvent({ type: "done" }, setMessages, vi.fn());
    expect(isTerminal).toBe(true);
    expect(messages[1].streaming).toBe(false);
  });

  it("returns true on cancelled, leaving partial text intact", () => {
    let messages = seedMessages();
    messages[1].text = "partial answer";
    const setMessages = (updater: any) => { messages = updater(messages); };
    const isTerminal = applyStreamEvent({ type: "cancelled" }, setMessages, vi.fn());
    expect(isTerminal).toBe(true);
    expect(messages[1].text).toBe("partial answer");
    expect(messages[1].streaming).toBe(false);
  });
});
