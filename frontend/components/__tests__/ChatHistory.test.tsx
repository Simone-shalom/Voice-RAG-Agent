import { act, render, screen, waitFor } from "@testing-library/react";
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

  it("requires a second click to actually delete (no native confirm dialog)", async () => {
    const user = userEvent.setup();
    render(
      <ChatHistory activeThreadId="t1" onSelectThread={vi.fn()} onNewChat={vi.fn()} refreshKey={0} />
    );

    await screen.findByText("Who is Patrick O'Neill?");
    await user.click(screen.getByLabelText(`Delete "Who is Patrick O'Neill?"`));

    // first click only arms the confirm state — nothing deleted yet
    expect(screen.getByText("Who is Patrick O'Neill?")).toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalledWith("/api/agent/threads/t2", { method: "DELETE" });

    await user.click(screen.getByLabelText(`Confirm delete "Who is Patrick O'Neill?"`));

    await waitFor(() => expect(screen.queryByText("Who is Patrick O'Neill?")).not.toBeInTheDocument());
    expect(fetch).toHaveBeenCalledWith("/api/agent/threads/t2", { method: "DELETE" });
  });

  it("cancels the pending delete without removing the thread", async () => {
    const user = userEvent.setup();
    render(
      <ChatHistory activeThreadId="t1" onSelectThread={vi.fn()} onNewChat={vi.fn()} refreshKey={0} />
    );

    await screen.findByText("Who is Patrick O'Neill?");
    await user.click(screen.getByLabelText(`Delete "Who is Patrick O'Neill?"`));
    await user.click(screen.getByLabelText(`Cancel deleting "Who is Patrick O'Neill?"`));

    expect(screen.getByText("Who is Patrick O'Neill?")).toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalledWith("/api/agent/threads/t2", { method: "DELETE" });
  });

  it("starts a new chat when the active thread is deleted", async () => {
    const user = userEvent.setup();
    const onNewChat = vi.fn();
    render(
      <ChatHistory activeThreadId="t1" onSelectThread={vi.fn()} onNewChat={onNewChat} refreshKey={0} />
    );

    await screen.findByText("What is the Orbital Edge program?");
    await user.click(screen.getByLabelText(`Delete "What is the Orbital Edge program?"`));
    await user.click(screen.getByLabelText(`Confirm delete "What is the Orbital Edge program?"`));

    expect(onNewChat).toHaveBeenCalledOnce();
  });

  it("auto-reverts the pending delete after the confirm window elapses", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <ChatHistory activeThreadId="t1" onSelectThread={vi.fn()} onNewChat={vi.fn()} refreshKey={0} />
    );

    await screen.findByText("Who is Patrick O'Neill?");
    await user.click(screen.getByLabelText(`Delete "Who is Patrick O'Neill?"`));
    expect(screen.getByLabelText(`Confirm delete "Who is Patrick O'Neill?"`)).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(3100);
    });

    expect(screen.getByLabelText(`Delete "Who is Patrick O'Neill?"`)).toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalledWith("/api/agent/threads/t2", { method: "DELETE" });
    vi.useRealTimers();
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
