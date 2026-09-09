const RECONNECT_DELAYS_MS = [500, 1500, 4000];

export function voiceStreamUrl(threadId: string): string {
  // NEXT_PUBLIC_WS_URL is the browser-facing var, read directly here.
  // NEXT_PUBLIC_API_URL is deliberately NOT read client-side elsewhere in
  // this codebase (see DECISIONS.md, Etap 14) because under
  // `docker compose up` it's set to the compose-internal DNS name
  // "backend", which no browser can resolve — it's only safe there
  // because next.config.ts's server-side rewrites() is the sole other
  // consumer. Deriving from it here as a fallback is still correct for
  // production (Vercel/Railway), where the backend has a real public
  // https:// URL and the http->ws swap is exactly right — it's only local
  // docker-compose dev that needs the explicit override.
  const wsBase = process.env.NEXT_PUBLIC_WS_URL;
  if (wsBase) return `${wsBase}/voice/stream?thread_id=${encodeURIComponent(threadId)}`;

  const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const derived = base.replace(/^http/, "ws");
  return `${derived}/voice/stream?thread_id=${encodeURIComponent(threadId)}`;
}

interface Handlers {
  onMessage: (event: Record<string, unknown>) => void;
  onOpen?: () => void;
  onClose?: () => void;
  // Called exactly once, instead of the final onClose, when the reconnect
  // budget (RECONNECT_DELAYS_MS) has been exhausted without ever
  // reopening — i.e. the connection is never coming back on its own.
  // Callers that need a fallback path (VoiceStream.tsx) should treat this
  // as "give up and fall back", not just another transient close.
  onGaveUp?: () => void;
}

export function connectVoiceStream(threadId: string, handlers: Handlers) {
  let socket: WebSocket | null = null;
  let attempt = 0;
  let deliberatelyClosed = false;
  let pendingTimerId: ReturnType<typeof setTimeout> | null = null;

  function open() {
    if (deliberatelyClosed) return;
    socket = new WebSocket(voiceStreamUrl(threadId));
    socket.onopen = () => {
      attempt = 0;
      handlers.onOpen?.();
    };
    socket.onmessage = (evt: MessageEvent) => {
      try {
        handlers.onMessage(JSON.parse(evt.data));
      } catch {
        // non-JSON frame — ignore
      }
    };
    socket.onclose = () => {
      handlers.onClose?.();
      if (deliberatelyClosed) return;
      if (attempt >= RECONNECT_DELAYS_MS.length) {
        handlers.onGaveUp?.();
        return;
      }
      const delay = RECONNECT_DELAYS_MS[attempt];
      attempt += 1;
      pendingTimerId = setTimeout(open, delay);
    };
  }

  open();

  return {
    send(data: Blob | string) {
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(data as never);
      }
      // otherwise: socket is CONNECTING/CLOSING/CLOSED — drop the chunk
      // rather than let WebSocket.send() throw InvalidStateError
      // synchronously inside a MediaRecorder event handler.
    },
    close() {
      deliberatelyClosed = true;
      if (pendingTimerId !== null) {
        clearTimeout(pendingTimerId);
        pendingTimerId = null;
      }
      socket?.close();
    },
  };
}
