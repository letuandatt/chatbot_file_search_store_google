"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/types/chat";
import { ThinkingSteps } from "./ThinkingSteps";

interface MessageListProps {
    messages: Message[];
    isLoading: boolean;
    hasActiveSession: boolean;
    expandedThinking: Set<number>;
    onToggleThinking: (idx: number) => void;
}

export function MessageList({
    messages,
    isLoading,
    hasActiveSession,
    expandedThinking,
    onToggleThinking,
}: MessageListProps) {
    const endRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    if (!hasActiveSession) {
        return (
            <div className="h-full flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 rounded-2xl bg-white flex items-center justify-center mb-6">
                    <svg
                        className="w-8 h-8 text-zinc-900"
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
                <h2 className="text-2xl font-bold mb-2">Chào mừng đến với LexMind!</h2>
                <p className="text-zinc-400 max-w-md leading-relaxed">
                    Tạo phiên làm việc mới <br /> hoặc chọn một phiên làm việc đã tồn tại.
                </p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="h-full flex items-center justify-center">
                <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full"></div>
            </div>
        );
    }

    return (
        <div className="max-w-5xl mx-auto space-y-6 px-4">
            {messages.map((msg, idx) => (
                <div key={idx} className="space-y-4">
                    {/* User Message */}
                    <div className="flex items-start gap-3 justify-end">
                        <div className="max-w-[80%] bg-zinc-700 text-white px-4 py-3 rounded-2xl rounded-tr-md">
                            <p className="text-sm leading-relaxed">{msg.question}</p>
                        </div>
                        <div className="w-8 h-8 rounded-full bg-zinc-600 flex items-center justify-center flex-shrink-0">
                            <svg
                                className="w-4 h-4 text-white"
                                fill="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                            </svg>
                        </div>
                    </div>

                    {msg.thinking_steps && msg.thinking_steps.length > 0 && (
                        <ThinkingSteps
                            steps={msg.thinking_steps}
                            expanded={expandedThinking.has(idx)}
                            onToggle={() => onToggleThinking(idx)}
                        />
                    )}

                    {/* Bot Response */}
                    <div className="flex items-start gap-3 justify-start">
                        <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center flex-shrink-0 shadow-md">
                            <svg
                                className="w-4 h-4 text-zinc-900"
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
                        <div className="max-w-[80%] bg-zinc-800 border border-zinc-700/50 text-zinc-100 px-4 py-3 rounded-2xl rounded-tl-md">
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                {msg.answer}
                            </p>
                        </div>
                    </div>
                </div>
            ))}
            {messages.length === 0 && (
                <div className="text-center py-16">
                    <div className="w-14 h-14 rounded-2xl bg-zinc-800 border border-zinc-700 flex items-center justify-center mx-auto mb-4">
                        <svg
                            className="w-7 h-7 text-zinc-500"
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
                    <p className="text-zinc-500 text-sm">Bắt đầu cuộc trò chuyện</p>
                    <p className="text-zinc-600 text-xs mt-1">Nhập câu hỏi của bạn ở bên dưới</p>
                </div>
            )}
            <div ref={endRef} />
        </div>
    );
}
