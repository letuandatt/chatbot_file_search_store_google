/**
 * Pure display formatters. Kept separate from API code so the
 * components can import them without dragging in fetch logic.
 */
import type { Session } from "@/types/chat";

export function formatSessionTitle(session: Session): string {
    if (session.title) return session.title;
    if (session.created_at) {
        const date = new Date(session.created_at);
        return date.toLocaleString("vi-VN", {
            day: "2-digit",
            month: "2-digit",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    }
    return "Phiên mới";
}

/** Truncate the basename of a filename, preserving the extension. */
export function formatFileName(name: string): string {
    const dot = name.lastIndexOf(".");
    if (dot === -1) {
        return name.length > 18 ? `${name.slice(0, 15)}...` : name;
    }
    const ext = name.slice(dot + 1);
    const baseName = name.slice(0, dot);
    if (baseName.length > 15) {
        return `${baseName.slice(0, 15)}...${ext}`;
    }
    return name;
}
