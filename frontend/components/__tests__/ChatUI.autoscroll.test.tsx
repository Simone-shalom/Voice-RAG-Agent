import { render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import ChatUI from "../ChatUI";

/**
 * jsdom never computes real layout — scrollHeight/clientHeight are always 0,
 * so an assertion like "did it scroll to the bottom" is meaningless without
 * faking those. Instead this stubs scrollHeight to a fixed "content is
 * taller than the viewport" value and spies on scrollTop, then verifies the
 * auto-scroll effect actually drives scrollTop to match scrollHeight as
 * streamed tokens arrive — i.e. the effect fires, the ref is attached to
 * the right element, and it recomputes on every `messages` update.
 * (Real browser behavior — no visible scrollbar, page itself doesn't
 * scroll — was verified manually against a live instance; see DECISIONS.md
 * Etap 15.)
 */

function stubScrollableContainer(container: HTMLElement) {
  const el = container.querySelector('[class*="overflow-y-auto"]') as HTMLDivElement;
  if (!el) throw new Error("scrollable message container not found");

  let scrollTopValue = 0;
  Object.defineProperty(el, "scrollHeight", { configurable: true, get: () => 900 });
  Object.defineProperty(el, "scrollTop", {
    configurable: true,
    get: () => scrollTopValue,
    set: (v: number) => {
      scrollTopValue = v;
    },
  });
  return {
    el,
    getScrollTop: () => scrollTopValue,
  };
}

function sseStreamResponse(events: object[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const e of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(e)}\n\n`));
      }
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (typeof url === "string" && url.includes("/api/agent/threads/")) {
        // simulates a brand new thread that has no history yet
        return Promise.resolve(new Response(null, { status: 404 }));
      }
      if (typeof url === "string" && url.includes("/api/agent/chat/stream")) {
        return Promise.resolve(
          sseStreamResponse([
            { type: "token", text: "This is a long streamed answer, " },
            { type: "token", text: "growing well past the viewport height " },
            { type: "token", text: "so the container actually needs to scroll." },
            { type: "done" },
          ])
        );
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    })
  );
});

describe("ChatUI auto-scroll", () => {
  it("scrolls the message container to the bottom as tokens stream in", async () => {
    const user = userEvent.setup();
    const { container, getByPlaceholderText } = render(
      <ChatUI selectedEpisode={null} episodes={[]} threadId="t1" />
    );

    const { getScrollTop } = stubScrollableContainer(container);

    const input = getByPlaceholderText("Type a question…") as HTMLInputElement;
    await user.type(input, "What is the Orbital Edge program?{Enter}");

    await waitFor(() => expect(getScrollTop()).toBe(900));
  });

  it("scrolls again on a second exchange in the same conversation", async () => {
    const user = userEvent.setup();
    const { container, getByPlaceholderText } = render(
      <ChatUI selectedEpisode={null} episodes={[]} threadId="t1" />
    );

    const { el, getScrollTop } = stubScrollableContainer(container);
    const input = getByPlaceholderText("Type a question…") as HTMLInputElement;

    await user.type(input, "First question{Enter}");
    await waitFor(() => expect(getScrollTop()).toBe(900));

    // simulate the user having scrolled up to re-read something
    el.scrollTop = 100;
    expect(getScrollTop()).toBe(100);

    await user.type(input, "Second question{Enter}");

    await waitFor(() => expect(getScrollTop()).toBe(900));
  });
});
