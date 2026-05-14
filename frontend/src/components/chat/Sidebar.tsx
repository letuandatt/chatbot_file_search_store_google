"use client";

import type { Session } from "@/types/chat";
import { formatSessionTitle } from "@/lib/formatters";

interface SidebarProps {
    collapsed: boolean;
    sessions: Session[];
    activeSession: string | null;
    editingSessionId: string | null;
    editingTitle: string;
    onCreateSession: () => void;
    onDeleteAllSessions: () => void;
    onSelectSession: (sessionId: string) => void;
    onStartRename: (sessionId: string, currentTitle: string) => void;
    onCommitRename: (sessionId: string) => void;
    onCancelRename: () => void;
    onChangeEditingTitle: (title: string) => void;
    onDeleteSession: (sessionId: string, e: React.MouseEvent) => void;
    onOpenProfile: () => void;
    onLogout: () => void;
    onDeleteAccount: () => void;
}

/**
 * Left rail: header (Tạo mới + Xóa tất cả), session list, footer
 * (Hồ sơ / Đăng xuất / Xóa tài khoản). All state lives in the
 * parent — the sidebar is "dumb", so we can swap it out for an
 * alternative layout (e.g. a topbar on mobile) later without
 * touching session logic.
 */
export function Sidebar({
    collapsed,
    sessions,
    activeSession,
    editingSessionId,
    editingTitle,
    onCreateSession,
    onDeleteAllSessions,
    onSelectSession,
    onStartRename,
    onCommitRename,
    onCancelRename,
    onChangeEditingTitle,
    onDeleteSession,
    onOpenProfile,
    onLogout,
    onDeleteAccount,
}: SidebarProps) {
    return (
        <div
            className={`${collapsed ? "w-0 overflow-hidden" : "w-64"} flex flex-col border-r border-zinc-800 transition-all duration-300`}
        >
            {/* Sidebar Header */}
            <div className="p-3 flex gap-2 border-b border-zinc-800">
                <button
                    onClick={onCreateSession}
                    className="flex-1 px-4 py-2.5 bg-white text-zinc-900 text-sm font-medium rounded-full hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 4v16m8-8H4"
                        />
                    </svg>
                    Tạo mới
                </button>
                <button
                    onClick={onDeleteAllSessions}
                    className="p-2.5 border border-zinc-700 hover:border-zinc-500 text-zinc-400 hover:text-white rounded-full transition-colors"
                    title="Xóa tất cả"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                    </svg>
                </button>
            </div>

            {/* Session List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {sessions.map((session) => (
                    <div
                        key={session.session_id}
                        className={`group flex items-center gap-3 px-3 py-3 cursor-pointer rounded-xl transition-all duration-200 ${
                            activeSession === session.session_id
                                ? "bg-zinc-800 border border-zinc-700"
                                : "hover:bg-zinc-900 border border-transparent"
                        }`}
                        onClick={() => onSelectSession(session.session_id)}
                    >
                        <div
                            className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                                activeSession === session.session_id
                                    ? "bg-white text-zinc-900"
                                    : "bg-zinc-800 text-zinc-500"
                            }`}
                        >
                            <svg
                                className="w-4 h-4"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={1.5}
                                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                                />
                            </svg>
                        </div>

                        {editingSessionId === session.session_id ? (
                            <input
                                type="text"
                                value={editingTitle}
                                onChange={(e) => onChangeEditingTitle(e.target.value)}
                                onBlur={() => onCommitRename(session.session_id)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter") onCommitRename(session.session_id);
                                    if (e.key === "Escape") onCancelRename();
                                }}
                                className="flex-1 bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-white"
                                autoFocus
                                onClick={(e) => e.stopPropagation()}
                            />
                        ) : (
                            <>
                                <div className="flex-1 min-w-0">
                                    <span
                                        className={`text-sm truncate block ${
                                            activeSession === session.session_id
                                                ? "text-white font-medium"
                                                : "text-zinc-400"
                                        }`}
                                    >
                                        {formatSessionTitle(session)}
                                    </span>
                                    {session.num_messages > 0 && (
                                        <span className="text-xs text-zinc-600">
                                            {session.num_messages} tin nhắn
                                        </span>
                                    )}
                                </div>
                                <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onStartRename(
                                                session.session_id,
                                                session.title || "",
                                            );
                                        }}
                                        className="p-1.5 hover:bg-zinc-700 rounded-lg transition-colors"
                                        title="Đổi tên"
                                    >
                                        <svg
                                            className="w-3.5 h-3.5 text-zinc-500 hover:text-zinc-300"
                                            fill="none"
                                            stroke="currentColor"
                                            viewBox="0 0 24 24"
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                strokeWidth={1.5}
                                                d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                                            />
                                        </svg>
                                    </button>
                                    <button
                                        onClick={(e) => onDeleteSession(session.session_id, e)}
                                        className="p-1.5 hover:bg-red-500/20 rounded-lg transition-colors"
                                        title="Xóa phiên"
                                    >
                                        <svg
                                            className="w-3.5 h-3.5 text-zinc-500 hover:text-red-400"
                                            fill="none"
                                            stroke="currentColor"
                                            viewBox="0 0 24 24"
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                strokeWidth={1.5}
                                                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                                            />
                                        </svg>
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                ))}
                {sessions.length === 0 && (
                    <div className="text-center py-12">
                        <div className="w-12 h-12 rounded-xl bg-zinc-800 flex items-center justify-center mx-auto mb-3">
                            <svg
                                className="w-6 h-6 text-zinc-600"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={1.5}
                                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                                />
                            </svg>
                        </div>
                        <p className="text-zinc-600 text-sm">Chưa có phiên nào</p>
                        <p className="text-zinc-700 text-xs mt-1">
                            Nhấn &quot;Tạo mới&quot; để bắt đầu
                        </p>
                    </div>
                )}
            </div>

            {/* Sidebar Footer */}
            <div className="p-3 border-t border-zinc-800 space-y-1">
                <button
                    onClick={onOpenProfile}
                    className="w-full px-3 py-2.5 text-sm text-zinc-400 hover:text-white hover:bg-zinc-800/50 rounded-xl transition-colors flex items-center gap-3"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                        />
                    </svg>
                    Hồ sơ
                </button>
                <button
                    onClick={onLogout}
                    className="w-full px-3 py-2.5 text-sm text-zinc-400 hover:text-white hover:bg-zinc-800/50 rounded-xl transition-colors flex items-center gap-3"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                        />
                    </svg>
                    Đăng xuất
                </button>
                <button
                    onClick={onDeleteAccount}
                    className="w-full px-3 py-2.5 text-sm text-red-400/80 hover:text-red-400 hover:bg-zinc-800/50 rounded-xl transition-colors flex items-center gap-3"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        />
                    </svg>
                    Xóa tài khoản
                </button>
            </div>
        </div>
    );
}
