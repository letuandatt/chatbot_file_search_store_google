/**
 * Domain endpoints for the chat page.
 *
 * Each function does one HTTP round-trip and returns either parsed
 * JSON or `null`. The page-level code then translates `null` into
 * "do nothing / show error toast" — keeping logging there means we
 * don't double-log inside this module.
 *
 * All functions accept an `onUnauthorized` callback so the caller
 * decides what to do on a 401 (the chat page wants a Next router
 * push; static pages can just let the wrapper redirect via
 * window.location).
 */

import {
    API_URL,
    apiFetch,
    authHeaders,
    authHeadersForFormData,
} from "@/lib/api";
import type { Message, Session, ThinkingStep } from "@/types/chat";

interface CallOptions {
    onUnauthorized?: () => void;
}

interface SessionDetail extends Session {
    messages: Message[];
}

export async function listSessions(opts: CallOptions = {}): Promise<Session[] | null> {
    const res = await apiFetch("/sessions/", { method: "GET" }, opts);
    if (!res.ok) return null;
    return res.json();
}

export async function createNewSession(opts: CallOptions = {}): Promise<Session | null> {
    const res = await apiFetch(
        "/sessions/",
        { method: "POST", body: JSON.stringify({ title: null }) },
        opts,
    );
    if (!res.ok) return null;
    return res.json();
}

export async function renameSessionApi(
    sessionId: string,
    newTitle: string,
    opts: CallOptions = {},
): Promise<boolean> {
    const res = await apiFetch(
        `/sessions/${sessionId}`,
        { method: "PUT", body: JSON.stringify({ title: newTitle }) },
        opts,
    );
    return res.ok;
}

export async function deleteSessionApi(
    sessionId: string,
    opts: CallOptions = {},
): Promise<boolean> {
    const res = await apiFetch(`/sessions/${sessionId}`, { method: "DELETE" }, opts);
    return res.ok || res.status === 204;
}

export async function deleteAllSessionsApi(opts: CallOptions = {}): Promise<boolean> {
    const res = await apiFetch("/sessions/", { method: "DELETE" }, opts);
    return res.ok;
}

export async function getSessionMessages(
    sessionId: string,
    opts: CallOptions = {},
): Promise<SessionDetail | null> {
    const res = await apiFetch(`/sessions/${sessionId}`, { method: "GET" }, opts);
    if (!res.ok) return null;
    return res.json();
}

interface ChatResponse {
    response: string;
    session_id?: string;
    thinking_steps?: ThinkingStep[];
}

/** Send a plain text message. */
export async function sendTextMessage(
    message: string,
    sessionId: string | null,
    opts: CallOptions = {},
): Promise<ChatResponse | null> {
    const res = await apiFetch(
        "/chat/text",
        {
            method: "POST",
            body: JSON.stringify({ message, session_id: sessionId }),
        },
        opts,
    );
    if (!res.ok) return null;
    return res.json();
}

/** Send a message with an attached image. */
export async function sendImageMessage(
    message: string,
    image: File,
    sessionId: string | null,
): Promise<ChatResponse | null> {
    const formData = new FormData();
    formData.append("message", message);
    formData.append("image", image);
    if (sessionId) formData.append("session_id", sessionId);

    const res = await fetch(`${API_URL}/chat/image`, {
        method: "POST",
        headers: authHeadersForFormData(),
        body: formData,
    });
    if (!res.ok) return null;
    return res.json();
}

interface FileUploadResponse {
    file_id: string;
    filename: string;
    status: string;
    message: string;
}

/** Upload a PDF for server-side processing. */
export async function uploadPdf(
    file: File,
    sessionId: string | null,
): Promise<FileUploadResponse | null> {
    const formData = new FormData();
    formData.append("file", file);
    if (sessionId) formData.append("session_id", sessionId);

    const res = await fetch(`${API_URL}/chat/upload`, {
        method: "POST",
        headers: authHeadersForFormData(),
        body: formData,
    });
    if (!res.ok) return null;
    return res.json();
}

interface FileStatusResponse {
    status: string;
}

export async function getFileStatus(
    fileId: string,
    opts: CallOptions = {},
): Promise<FileStatusResponse | null> {
    const res = await apiFetch(`/chat/file/${fileId}/status`, { method: "GET" }, opts);
    if (!res.ok) return null;
    return res.json();
}

export async function deleteAccountApi(opts: CallOptions = {}): Promise<boolean> {
    const res = await apiFetch(`/auth/account`, { method: "DELETE" }, opts);
    return res.ok;
}

// Re-export for compatibility with existing callers that imported
// these from page.tsx.
export { authHeaders };
