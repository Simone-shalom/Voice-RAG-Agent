import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import ChatHistory from "../ChatHistory";

const THREADS = [
  { id: "t1", title: "What is the Orbital Edge program?", updated_at: "2026-09-08T12:00:00" },
  { id: "t2", title: "Who is Patrick O'Neill?", updated_at: "2026-09-08T11:00:00" },
];

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response(JSON.stringify(THREADS), { status: 200 })))
  );
});

describe("ChatHistory", () => {
  it("lists threads from the backend", async () => {
    render(
      <ChatHistory activeThreadId="t1" onSelectThread={vi.fn()} onNewChat={vi.fn()} refreshKey={0} />
    );

    expect(await screen.findByText("What is the Orbital Edge program?")).toBeInTheDocument();
    expect(screen.getByText("Who is Patrick O'Neill?")).toBeInTheDocument();
  });

  it("calls onSelectThread when a past conversation is clicked", async () => {
    const onSelectThread = vi.fn();
    const user = userEvent.setup();
    render(
      <ChatHistory activeThreadId="t1" onSelectThread={onSelectThread} onNewChat={vi.fn()} refreshKey={0} />
    );

    const item = await screen.findByText("Who is Patrick O'Neill?");
    await user.click(item);

    expect(onSelectThread).toHaveBeenCalledWith("t2");
  });

  it("calls onNewChat when '+ New' is clicked", async () => {
    const onNewChat = vi.fn();
    const user = userEvent.setup();
    render(
      <ChatHistory activeThreadId="t1" onSelectThread={vi.fn()} onNewChat={onNewChat} refreshKey={0} />
    );

    await user.click(screen.getByText("+ New"));

    expect(onNewChat).toHaveBeenCalledOnce();
  });

  it("shows an unsaved placeholder when the active thread has no history yet", async () => {
    render(
      <ChatHistory activeThreadId="brand-new-id" onSelectThread={vi.fn()} onNewChat={vi.fn()} refreshKey={0} />
    );

    await waitFor(() => expect(screen.getByText("New conversation")).toBeInTheDocument());
    expect(screen.getByText("not sent yet")).toBeInTheDocument();
  });

  it("refetches when refreshKey changes", async () => {
    const { rerender } = render(
      <ChatHistory activeThreadId="t1" onSelectThread={vi.fn()} onNewChat={vi.fn()} refreshKey={0} />
    );
    await screen.findByText("What is the Orbital Edge program?");
    expect(fetch).toHaveBeenCalledTimes(1);

    rerender(
      <ChatHistory activeThreadId="t1" onSelectThread={vi.fn()} onNewChat={vi.fn()} refreshKey={1} />
    );

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  });
});
