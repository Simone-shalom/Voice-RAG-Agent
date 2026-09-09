import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import ChatUI from "../ChatUI";

// Fix 5: ChatUI's typed-send path and VoiceStream's voice-turn path each
// independently mutate "the last assistant message" — without a shared
// guard, a typed send while a voice answer is streaming (or vice versa)
// can interleave tokens into the same bubble. This proves ChatUI disables
// its own send path (and refuses to call the chat-stream endpoint) while
// VoiceStream reports a voice turn in progress.

vi.mock("../../lib/voiceWebSocket", () => ({
  connectVoiceStream: vi.fn(),
}));

import { connectVoiceStream } from "../../lib/voiceWebSocket";

let chatStreamCalls = 0;

beforeEach(() => {
  chatStreamCalls = 0;
  (globalThis.navigator.mediaDevices as any).getUserMedia = vi.fn().mockResolvedValue({
    getTracks: () => [{ stop: vi.fn() }],
  });

  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (typeof url === "string" && url.includes("/api/agent/threads/")) {
        return Promise.resolve(new Response(null, { status: 404 }));
      }
      if (typeof url === "string" && url.includes("/api/agent/chat/stream")) {
        chatStreamCalls += 1;
        return Promise.resolve(new Response(null, { status: 200 }));
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    })
  );
});

describe("ChatUI + VoiceStream shared answering guard", () => {
  it("disables the typed-send path while a voice turn is actively streaming", async () => {
    let voiceHandlers: any;
    (connectVoiceStream as any).mockImplementation((_threadId: string, handlers: any) => {
      voiceHandlers = handlers;
      handlers.onOpen?.();
      return { send: vi.fn(), close: vi.fn() };
    });

    render(<ChatUI selectedEpisode={null} episodes={[]} threadId="t1" />);

    const input = (await screen.findByPlaceholderText("Type a question…")) as HTMLInputElement;
    const sendButton = screen.getByRole("button", { name: "Send" });
    expect(input).not.toBeDisabled();

    // Start a voice turn — first token of a new turn, mirroring what
    // VoiceStream.tsx's onMessage handler does.
    voiceHandlers.onMessage({ type: "token", text: "Hi" });

    await waitFor(() => expect(input).toBeDisabled());
    expect(sendButton).toBeDisabled();

    // Clicking a disabled Send button is a normal DOM no-op (React never
    // fires the handler) — confirms the UI-level guard actually blocks
    // interaction, not just that a boolean flipped somewhere.
    sendButton.click();
    expect(chatStreamCalls).toBe(0);

    // Turn ends — guard releases and typed input is usable again.
    voiceHandlers.onMessage({ type: "done" });
    await waitFor(() => expect(input).not.toBeDisabled());
    expect(sendButton).not.toBeDisabled();
  });
});
