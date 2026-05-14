"use client";

interface ChatHeaderProps {
    onToggleSidebar: () => void;
    onResetSession: () => void;
    onOpenFiles: () => void;
}

export function ChatHeader({ onToggleSidebar, onResetSession, onOpenFiles }: ChatHeaderProps) {
    return (
        <div className="h-14 px-4 flex items-center justify-between border-b border-zinc-800">
            <div className="flex items-center gap-3">
                <button
                    onClick={onToggleSidebar}
                    className="p-2 hover:bg-zinc-800 rounded-xl transition-colors"
                    aria-label="Mở/đóng sidebar"
                >
                    <svg
                        className="w-5 h-5 text-zinc-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M4 6h16M4 12h16M4 18h16"
                        />
                    </svg>
                </button>
                <button
                    onClick={onResetSession}
                    className="flex items-center gap-2 hover:opacity-80 transition-opacity"
                >
                    <div className="w-8 h-8 rounded-xl bg-white flex items-center justify-center">
                        <svg
                            className="w-5 h-5 text-zinc-900"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={1.5}
                                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                            />
                        </svg>
                    </div>
                    <span className="text-lg font-semibold">LexMind</span>
                </button>
            </div>
            <div className="flex items-center gap-2">
                <button
                    onClick={onOpenFiles}
                    className="w-9 h-9 flex items-center justify-center hover:bg-zinc-800 rounded-xl transition-colors"
                    title="File đã tải"
                >
                    <svg className="w-5 h-5 text-amber-400" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z" />
                    </svg>
                </button>
            </div>
        </div>
    );
}
