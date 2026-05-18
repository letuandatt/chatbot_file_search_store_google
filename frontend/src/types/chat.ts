/**
 * Shared chat types. Mirrors the Pydantic models in
 * `backend/models/chat.py`. Kept here (not co-located with components)
 * so non-React consumers (e.g. tests, future SSE clients) can import
 * them without touching the React tree.
 */

export interface Session {
    session_id: string;
    title: string | null;
    created_at: string | null;
    updated_at: string | null;
    num_messages: number;
}

export interface ThinkingStep {
    agent: string;
    action: string;
    detail?: string;
}

export interface Message {
    question: string;
    answer: string;
    timestamp: string;
    thinking_steps?: ThinkingStep[];
}

/** Status of an in-flight or completed upload, as tracked client-side. */
export type UploadedFileStatus =
    | "uploading"
    | "uploaded"
    | "processed"
    | "ready"
    | "error";

export interface UploadedFile {
    name: string;
    status: UploadedFileStatus;
    /** Distinguishes PDF (server-side processing pipeline) vs image (sent inline). */
    fileType: "pdf" | "image";
    /** Mongo document id; only present after the upload returns. */
    fileId?: string;
    /** Kept around for images so we can re-send via FormData on submit. */
    rawFile?: File;
}

/** A file is usable in a query when it's either fully processed (PDF)
 *  or queued in memory waiting to be sent (image). */
export function isFileReady(file: UploadedFile | null): boolean {
    if (!file) return true;
    return file.status === "processed" || file.status === "ready";
}
