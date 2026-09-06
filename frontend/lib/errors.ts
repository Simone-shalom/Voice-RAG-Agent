/**
 * Extracts a clean, user-facing message from a failed fetch Response.
 * Backend errors are JSON `{"detail": "..."}` (see CLAUDE.md's error-handling
 * convention), except rate-limit responses from slowapi, which use
 * `{"error": "..."}` instead — check both before falling back to a generic
 * status-code message for anything else (an HTML error page, a stack trace,
 * an empty body) so raw internals never reach the UI.
 */
export async function friendlyErrorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (typeof body?.error === "string") return body.error;
  } catch {
    // not JSON — fall through to a generic message
  }
  return `Request failed (${res.status})`;
}
