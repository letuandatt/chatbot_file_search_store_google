"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Session {
    session_id: string;
    title: string | null;
    created_at: string | null;
    updated_at: string | null;
    num_messages: number;
}

interface ThinkingStep {
    agent: string;
    action: string;
    detail?: string;
}

interface Message {
    question: string;
    answer: string;
    timestamp: string;
    thinking_steps?: ThinkingStep[];
}

export default function ChatPage() {
    const router = useRouter();
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // State
    const [sessions, setSessions] = useState<Session[]>([]);
    const [activeSession, setActiveSession] = useState<string | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputMessage, setInputMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [uploadingFile, setUploadingFile] = useState(false);
    const [uploadedFile, setUploadedFile] = useState<{
        name: string,
        status: "uploading" | "uploaded" | "processed" | "ready" | "error",
        fileType: "pdf" | "image",
        fileId?: string,
        rawFile?: File
    } | null>(null);
    const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
    const [editingTitle, setEditingTitle] = useState("");
    const [expandedThinking, setExpandedThinking] = useState<Set<number>>(new Set());

    // Auth is now driven by httpOnly cookies set on /auth/login; the
    // frontend never reads or stores the token itself, so every fetch
    // must include credentials. JSON endpoints still need a Content-Type;
    // multipart endpoints (uploads, image chat) must NOT set one so the
    // browser fills in the multipart boundary automatically.
    const jsonHeaders = { "Content-Type": "application/json" } as const;
    const handleAuthFailure = () => {
        router.push("/login");
    };

    // Check auth on mount
    useEffect(() => {
        loadSessions();
    }, []);

    // Scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Load sessions
    const loadSessions = async () => {
        try {
            const res = await fetch(`${API_URL}/sessions/`, {
                credentials: "include",
            });
            if (res.status === 401) {
                handleAuthFailure();
                return;
            }
            if (res.ok) {
                const data = await res.json();
                setSessions(data);
            }
        } catch (error) {
            console.error("Error loading sessions:", error);
        }
    };

    // Create new session
    const createSession = async () => {
        try {
            const res = await fetch(`${API_URL}/sessions/`, {
                method: "POST",
                credentials: "include",
                headers: jsonHeaders,
                body: JSON.stringify({ title: null }),
            });
            if (res.ok) {
                const newSession = await res.json();
                setSessions([newSession, ...sessions]);
                setActiveSession(newSession.session_id);
                setMessages([]);
            }
        } catch (error) {
            console.error("Error creating session:", error);
        }
    };

    // Rename session
    const renameSession = async (sessionId: string, newTitle: string) => {
        if (!newTitle.trim()) {
            setEditingSessionId(null);
            return;
        }
        try {
            const res = await fetch(`${API_URL}/sessions/${sessionId}`, {
                method: "PUT",
                credentials: "include",
                headers: jsonHeaders,
                body: JSON.stringify({ title: newTitle.trim() }),
            });
            if (res.ok) {
                setSessions(sessions.map(s =>
                    s.session_id === sessionId ? { ...s, title: newTitle.trim() } : s
                ));
            }
        } catch (error) {
            console.error("Error renaming session:", error);
        } finally {
            setEditingSessionId(null);
        }
    };

    // Delete session
    const deleteSession = async (sessionId: string, e?: React.MouseEvent) => {
        e?.stopPropagation();
        if (!confirm("Xóa phiên làm việc này?")) return;
        try {
            const res = await fetch(`${API_URL}/sessions/${sessionId}`, {
                method: "DELETE",
                credentials: "include",
            });
            if (res.ok || res.status === 204) {
                setSessions(sessions.filter(s => s.session_id !== sessionId));
                if (activeSession === sessionId) {
                    setActiveSession(null);
                    setMessages([]);
                }
            }
        } catch (error) {
            console.error("Error deleting session:", error);
        }
    };

    // Delete all sessions
    const deleteAllSessions = async () => {
        if (!confirm("Bạn có chắc muốn xóa tất cả phiên làm việc?")) return;
        try {
            const res = await fetch(`${API_URL}/sessions/`, {
                method: "DELETE",
                credentials: "include",
            });
            if (res.ok) {
                setSessions([]);
                setActiveSession(null);
                setMessages([]);
            }
        } catch (error) {
            console.error("Error deleting all sessions:", error);
        }
    };

    // Load session messages
    const loadSessionMessages = async (sessionId: string) => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_URL}/sessions/${sessionId}`, {
                credentials: "include",
            });
            if (res.ok) {
                const data = await res.json();
                // Map messages to include thinking_steps if present
                const loadedMessages = (data.messages || []).map((msg: Message) => ({
                    ...msg,
                    thinking_steps: msg.thinking_steps || undefined,
                }));
                setMessages(loadedMessages);
                setActiveSession(sessionId);
            }
        } catch (error) {
            console.error("Error loading messages:", error);
        } finally {
            setIsLoading(false);
        }
    };

    // Send message (text or with image)
    const sendMessage = async () => {
        if (!inputMessage.trim() || isSending) return;

        const userMessage = inputMessage.trim();
        setInputMessage("");
        setIsSending(true);

        const tempMessage: Message = {
            question: userMessage,
            answer: "Đang xử lý...",
            timestamp: new Date().toISOString(),
        };
        setMessages([...messages, tempMessage]);

        try {
            let res;

            if (uploadedFile?.fileType === "image" && uploadedFile.rawFile) {
                // Send with image via /chat/image endpoint
                const formData = new FormData();
                formData.append("message", userMessage);
                formData.append("image", uploadedFile.rawFile);
                if (activeSession) {
                    formData.append("session_id", activeSession);
                }

                res = await fetch(`${API_URL}/chat/image`, {
                    method: "POST",
                    credentials: "include",
                    body: formData,
                });

                // Clear image after sending
                setUploadedFile(null);
            } else {
                // Regular text message via /chat/text endpoint
                res = await fetch(`${API_URL}/chat/text`, {
                    method: "POST",
                    credentials: "include",
                    headers: jsonHeaders,
                    body: JSON.stringify({
                        message: userMessage,
                        session_id: activeSession,
                    }),
                });
            }

            if (res.ok) {
                const data = await res.json();
                setMessages(prev => {
                    const updated = [...prev];
                    updated[updated.length - 1] = {
                        question: userMessage,
                        answer: data.response,
                        timestamp: new Date().toISOString(),
                        thinking_steps: data.thinking_steps || undefined,
                    };
                    return updated;
                });

                if (!activeSession && data.session_id) {
                    setActiveSession(data.session_id);
                    loadSessions();
                }
            } else {
                throw new Error("Failed to send message");
            }
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1].answer = "Lỗi khi gửi tin nhắn. Vui lòng thử lại.";
                return updated;
            });
        } finally {
            setIsSending(false);
        }
    };

    // Handle file upload - different flow for images vs PDFs
    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const isImage = file.type.startsWith('image/');
        const isPdf = file.name.toLowerCase().endsWith('.pdf');

        if (!isImage && !isPdf) {
            alert("Chỉ hỗ trợ file ảnh (jpg, png, gif, webp) hoặc PDF.");
            return;
        }

        if (isImage) {
            // Images are ready to use immediately - store the raw file
            setUploadedFile({
                name: file.name,
                status: "ready",
                fileType: "image",
                rawFile: file
            });
            e.target.value = "";
            return;
        }

        // PDF flow - upload and wait for processing
        setUploadingFile(true);
        setUploadedFile({ name: file.name, status: "uploading", fileType: "pdf" });

        const formData = new FormData();
        formData.append("file", file);
        if (activeSession) {
            formData.append("session_id", activeSession);
        }

        try {
            const res = await fetch(`${API_URL}/chat/upload`, {
                method: "POST",
                credentials: "include",
                body: formData,
            });
            if (res.ok) {
                const data = await res.json();
                setUploadedFile({ name: file.name, status: "uploaded", fileType: "pdf", fileId: data.file_id });
                // Start polling for status
                pollFileStatus(data.file_id, file.name);
            } else {
                throw new Error("Upload failed");
            }
        } catch (error) {
            console.error("Error uploading file:", error);
            alert("Lỗi khi tải file lên.");
            setUploadedFile(null);
        } finally {
            setUploadingFile(false);
            e.target.value = "";
        }
    };

    // Poll PDF file status from database
    const pollFileStatus = async (fileId: string, fileName: string) => {
        const maxAttempts = 60; // 2 minutes max
        let attempts = 0;

        const checkStatus = async () => {
            try {
                const res = await fetch(`${API_URL}/chat/file/${fileId}/status`, {
                    credentials: "include",
                });
                if (res.ok) {
                    const data = await res.json();
                    if (data.status === "processed") {
                        setUploadedFile({ name: fileName, status: "processed", fileType: "pdf", fileId });
                        return;
                    } else if (data.status === "error_processing") {
                        setUploadedFile({ name: fileName, status: "error", fileType: "pdf", fileId });
                        alert("File processing failed.");
                        return;
                    }
                }
                // Continue polling
                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(checkStatus, 2000); // Poll every 2 seconds
                } else {
                    alert("File processing timeout.");
                    setUploadedFile(null);
                }
            } catch (error) {
                console.error("Error polling file status:", error);
            }
        };

        // Start polling after a short delay
        setTimeout(checkStatus, 1000);
    };

    // Clear uploaded file
    const clearUploadedFile = () => {
        setUploadedFile(null);
    };

    // Logout
    const handleLogout = async () => {
        if (!confirm("Are u sure you want to logout?")) return;
        try {
            await fetch(`${API_URL}/auth/logout`, {
                method: "POST",
                credentials: "include",
            });
        } catch (error) {
            console.error("Error logging out:", error);
        }
        router.push("/login");
    };

    // Delete Account
    const handleDeleteAccount = async () => {
        if (!confirm("Bạn có chắc chắn muốn XÓA TÀI KHOẢN? Hành động này không thể hoàn tác!")) return;
        try {
            const res = await fetch(`${API_URL}/auth/account`, {
                method: "DELETE",
                credentials: "include",
            });
            if (res.ok) {
                alert("Tài khoản đã được xóa thành công.");
                router.push("/");
            } else {
                throw new Error("Delete failed");
            }
        } catch (error) {
            console.error("Error deleting account:", error);
            alert("Lỗi khi xóa tài khoản.");
        }
    };

    // Format session title
    const formatSessionTitle = (session: Session) => {
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
    };

    // Format filename for display
    const formatFileName = (name: string) => {
        const ext = name.split('.').pop() || '';
        const baseName = name.slice(0, name.lastIndexOf('.'));
        if (baseName.length > 15) {
            return `${baseName.slice(0, 15)}...${ext}`;
        }
        return name;
    };

    return (
        <div className="h-screen flex bg-zinc-950 text-white">
            {/* Sidebar */}
            <div className={`${sidebarCollapsed ? "w-0 overflow-hidden" : "w-64"} flex flex-col border-r border-zinc-800 transition-all duration-300`}>
                {/* Sidebar Header */}
                <div className="p-3 flex gap-2 border-b border-zinc-800">
                    <button
                        onClick={createSession}
                        className="flex-1 px-4 py-2.5 bg-white text-zinc-900 text-sm font-medium rounded-full hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        Tạo mới
                    </button>
                    <button
                        onClick={deleteAllSessions}
                        className="p-2.5 border border-zinc-700 hover:border-zinc-500 text-zinc-400 hover:text-white rounded-full transition-colors"
                        title="Xóa tất cả"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                    </button>
                </div>

                {/* Session List */}
                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                    {sessions.map((session) => (
                        <div
                            key={session.session_id}
                            className={`group flex items-center gap-3 px-3 py-3 cursor-pointer rounded-xl transition-all duration-200 ${activeSession === session.session_id
                                ? "bg-zinc-800 border border-zinc-700"
                                : "hover:bg-zinc-900 border border-transparent"
                                }`}
                            onClick={() => loadSessionMessages(session.session_id)}
                        >
                            {/* Session Icon */}
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${activeSession === session.session_id
                                ? "bg-white text-zinc-900"
                                : "bg-zinc-800 text-zinc-500"
                                }`}>
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                </svg>
                            </div>

                            {editingSessionId === session.session_id ? (
                                <input
                                    type="text"
                                    value={editingTitle}
                                    onChange={(e) => setEditingTitle(e.target.value)}
                                    onBlur={() => renameSession(session.session_id, editingTitle)}
                                    onKeyDown={(e) => {
                                        if (e.key === "Enter") renameSession(session.session_id, editingTitle);
                                        if (e.key === "Escape") setEditingSessionId(null);
                                    }}
                                    className="flex-1 bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-white"
                                    autoFocus
                                    onClick={(e) => e.stopPropagation()}
                                />
                            ) : (
                                <>
                                    <div className="flex-1 min-w-0">
                                        <span className={`text-sm truncate block ${activeSession === session.session_id
                                            ? "text-white font-medium"
                                            : "text-zinc-400"
                                            }`}>
                                            {formatSessionTitle(session)}
                                        </span>
                                        {session.num_messages > 0 && (
                                            <span className="text-xs text-zinc-600">
                                                {session.num_messages} tin nhắn
                                            </span>
                                        )}
                                    </div>
                                    {/* Action buttons */}
                                    <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setEditingSessionId(session.session_id);
                                                setEditingTitle(session.title || "");
                                            }}
                                            className="p-1.5 hover:bg-zinc-700 rounded-lg transition-colors"
                                            title="Đổi tên"
                                        >
                                            <svg className="w-3.5 h-3.5 text-zinc-500 hover:text-zinc-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                                            </svg>
                                        </button>
                                        <button
                                            onClick={(e) => deleteSession(session.session_id, e)}
                                            className="p-1.5 hover:bg-red-500/20 rounded-lg transition-colors"
                                            title="Xóa phiên"
                                        >
                                            <svg className="w-3.5 h-3.5 text-zinc-500 hover:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
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
                                <svg className="w-6 h-6 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                </svg>
                            </div>
                            <p className="text-zinc-600 text-sm">Chưa có phiên nào</p>
                            <p className="text-zinc-700 text-xs mt-1">Nhấn "Tạo mới" để bắt đầu</p>
                        </div>
                    )}
                </div>

                {/* Sidebar Footer */}
                <div className="p-3 border-t border-zinc-800 space-y-1">
                    <button
                        onClick={() => router.push("/profile")}
                        className="w-full px-3 py-2.5 text-sm text-zinc-400 hover:text-white hover:bg-zinc-800/50 rounded-xl transition-colors flex items-center gap-3"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                        Hồ sơ
                    </button>
                    <button
                        onClick={handleLogout}
                        className="w-full px-3 py-2.5 text-sm text-zinc-400 hover:text-white hover:bg-zinc-800/50 rounded-xl transition-colors flex items-center gap-3"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                        </svg>
                        Đăng xuất
                    </button>
                    <button
                        onClick={handleDeleteAccount}
                        className="w-full px-3 py-2.5 text-sm text-red-400/80 hover:text-red-400 hover:bg-zinc-800/50 rounded-xl transition-colors flex items-center gap-3"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        Xóa tài khoản
                    </button>
                </div>
            </div>

            {/* Main Area */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="h-14 px-4 flex items-center justify-between border-b border-zinc-800">
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                            className="p-2 hover:bg-zinc-800 rounded-xl transition-colors"
                        >
                            <svg className="w-5 h-5 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                        <button
                            onClick={() => { setActiveSession(null); setMessages([]); }}
                            className="flex items-center gap-2 hover:opacity-80 transition-opacity"
                        >
                            <div className="w-8 h-8 rounded-xl bg-white flex items-center justify-center">
                                <svg className="w-5 h-5 text-zinc-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                </svg>
                            </div>
                            <span className="text-lg font-semibold">LexMind</span>
                        </button>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => router.push("/files")}
                            className="w-9 h-9 flex items-center justify-center hover:bg-zinc-800 rounded-xl transition-colors"
                            title="File đã tải"
                        >
                            <svg className="w-5 h-5 text-amber-400" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z" />
                            </svg>
                        </button>
                    </div>
                </div>

                {/* Chat Area */}
                <div className="flex-1 overflow-y-auto p-6">
                    {!activeSession ? (
                        <div className="h-full flex flex-col items-center justify-center text-center">
                            <div className="w-16 h-16 rounded-2xl bg-white flex items-center justify-center mb-6">
                                <svg className="w-8 h-8 text-zinc-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                </svg>
                            </div>
                            <h2 className="text-2xl font-bold mb-2">Chào mừng đến với LexMind!</h2>
                            <p className="text-zinc-400 max-w-md leading-relaxed">
                                Tạo phiên làm việc mới <br /> hoặc chọn một phiên làm việc đã tồn tại.
                            </p>
                        </div>
                    ) : isLoading ? (
                        <div className="h-full flex items-center justify-center">
                            <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full"></div>
                        </div>
                    ) : (
                        <div className="max-w-5xl mx-auto space-y-6 px-4">
                            {messages.map((msg, idx) => (
                                <div key={idx} className="space-y-4">
                                    {/* User Message */}
                                    <div className="flex items-start gap-3 justify-end">
                                        <div className="max-w-[80%] bg-zinc-700 text-white px-4 py-3 rounded-2xl rounded-tr-md">
                                            <p className="text-sm leading-relaxed">{msg.question}</p>
                                        </div>
                                        <div className="w-8 h-8 rounded-full bg-zinc-600 flex items-center justify-center flex-shrink-0">
                                            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                                                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                                            </svg>
                                        </div>
                                    </div>

                                    {/* Thinking Box - Timeline Style */}
                                    {msg.thinking_steps && msg.thinking_steps.length > 0 && (
                                        <div className="flex items-start gap-3 justify-start ml-11 mb-2">
                                            <div className="w-full max-w-[85%]">
                                                <button
                                                    onClick={() => {
                                                        const newSet = new Set(expandedThinking);
                                                        if (newSet.has(idx)) {
                                                            newSet.delete(idx);
                                                        } else {
                                                            newSet.add(idx);
                                                        }
                                                        setExpandedThinking(newSet);
                                                    }}
                                                    className="group flex items-center gap-2 text-xs text-violet-400 hover:text-violet-300 transition-all duration-200 mb-2 py-1 px-2 -ml-2 rounded-lg hover:bg-violet-500/10"
                                                >
                                                    <svg
                                                        className={`w-3.5 h-3.5 transition-transform duration-200 ${expandedThinking.has(idx) ? 'rotate-90' : ''}`}
                                                        fill="currentColor"
                                                        viewBox="0 0 20 20"
                                                    >
                                                        <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                                                    </svg>
                                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                                    </svg>
                                                    <span className="font-medium">Quá trình suy luận</span>
                                                    <span className="text-violet-500/70 text-[10px] bg-violet-500/20 px-1.5 py-0.5 rounded-full">{msg.thinking_steps.length}</span>
                                                </button>

                                                {/* Timeline Content */}
                                                <div className={`overflow-hidden transition-all duration-300 ease-in-out ${expandedThinking.has(idx) ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'}`}>
                                                    <div className="bg-gradient-to-br from-violet-500/5 to-violet-600/10 border border-violet-500/20 rounded-xl p-4">
                                                        <div className="relative">
                                                            {/* Vertical timeline line */}
                                                            <div className="absolute left-[11px] top-3 bottom-3 w-0.5 bg-gradient-to-b from-violet-500/50 via-violet-400/30 to-violet-500/10 rounded-full"></div>

                                                            {/* Steps */}
                                                            <div className="space-y-3">
                                                                {msg.thinking_steps.map((step, stepIdx) => (
                                                                    <div key={stepIdx} className="flex items-start gap-3 relative">
                                                                        {/* Step circle */}
                                                                        <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 z-10 transition-all duration-200
                                                                            ${stepIdx === msg.thinking_steps!.length - 1
                                                                                ? 'bg-violet-500 shadow-lg shadow-violet-500/30'
                                                                                : 'bg-violet-500/30 border border-violet-400/50'}`}
                                                                        >
                                                                            {stepIdx === msg.thinking_steps!.length - 1 ? (
                                                                                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                                                                                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                                                </svg>
                                                                            ) : (
                                                                                <span className="text-[10px] text-violet-300 font-semibold">{stepIdx + 1}</span>
                                                                            )}
                                                                        </div>

                                                                        {/* Step content */}
                                                                        <div className="flex-1 min-w-0 pb-1">
                                                                            <div className="flex items-center gap-2 flex-wrap">
                                                                                <span className="text-xs font-semibold text-violet-300 bg-violet-500/20 px-2 py-0.5 rounded-md">
                                                                                    {step.agent}
                                                                                </span>
                                                                            </div>
                                                                            <p className="text-xs text-zinc-400 mt-1 leading-relaxed">{step.action}</p>
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Bot Response */}
                                    <div className="flex items-start gap-3 justify-start">
                                        <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center flex-shrink-0 shadow-md">
                                            <svg className="w-4 h-4 text-zinc-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                            </svg>
                                        </div>
                                        <div className="max-w-[80%] bg-zinc-800 border border-zinc-700/50 text-zinc-100 px-4 py-3 rounded-2xl rounded-tl-md">
                                            <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.answer}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                            {messages.length === 0 && (
                                <div className="text-center py-16">
                                    <div className="w-14 h-14 rounded-2xl bg-zinc-800 border border-zinc-700 flex items-center justify-center mx-auto mb-4">
                                        <svg className="w-7 h-7 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                        </svg>
                                    </div>
                                    <p className="text-zinc-500 text-sm">Bắt đầu cuộc trò chuyện</p>
                                    <p className="text-zinc-600 text-xs mt-1">Nhập câu hỏi của bạn ở bên dưới</p>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Input Area */}
                {activeSession && (
                    <div className="p-4 border-t border-zinc-800">
                        <div className="max-w-4xl mx-auto flex items-center gap-3">
                            {/* Uploaded file indicator - moved to left */}
                            {uploadedFile && (
                                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs ${(uploadedFile.status === "processed" || uploadedFile.status === "ready") ? 'bg-emerald-500/20 text-emerald-400' :
                                    uploadedFile.status === "error" ? 'bg-red-500/20 text-red-400' :
                                        'bg-amber-500/20 text-amber-400'
                                    }`}>
                                    {uploadedFile.status !== "processed" && uploadedFile.status !== "ready" && uploadedFile.status !== "error" ? (
                                        <div className="animate-spin w-3 h-3 border border-current border-t-transparent rounded-full"></div>
                                    ) : (uploadedFile.status === "processed" || uploadedFile.status === "ready") ? (
                                        <>
                                            {uploadedFile.fileType === "image" ? (
                                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                                                </svg>
                                            ) : (
                                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                </svg>
                                            )}
                                        </>
                                    ) : (
                                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                        </svg>
                                    )}
                                    <span>{formatFileName(uploadedFile.name)}</span>
                                    <button onClick={clearUploadedFile} className="hover:text-white ml-1">
                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                        </svg>
                                    </button>
                                </div>
                            )}

                            <label className="p-3 hover:bg-zinc-800 rounded-xl cursor-pointer transition-colors">
                                <input
                                    type="file"
                                    accept=".pdf,image/*"
                                    className="hidden"
                                    onChange={handleFileUpload}
                                    disabled={uploadingFile || (uploadedFile !== null && !["processed", "ready"].includes(uploadedFile.status))}
                                />
                                {uploadingFile ? (
                                    <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                                ) : (
                                    <svg className="w-5 h-5 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                                    </svg>
                                )}
                            </label>

                            <input
                                type="text"
                                value={inputMessage}
                                onChange={(e) => setInputMessage(e.target.value)}
                                onKeyPress={(e) => e.key === "Enter" && !(uploadedFile !== null && !["processed", "ready"].includes(uploadedFile.status)) && sendMessage()}
                                placeholder={(uploadedFile !== null && !["processed", "ready"].includes(uploadedFile.status)) ? "Đang xử lý file..." : uploadedFile?.fileType === "image" ? "Hỏi về ảnh này..." : "Gõ câu hỏi..."}
                                className="flex-1 h-12 px-4 bg-zinc-800/50 border border-zinc-700 rounded-xl text-white placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none transition-colors"
                                disabled={isSending || (uploadedFile !== null && !["processed", "ready"].includes(uploadedFile.status))}
                            />

                            <button
                                onClick={sendMessage}
                                disabled={!inputMessage.trim() || isSending || (uploadedFile !== null && !["processed", "ready"].includes(uploadedFile.status))}
                                className="px-5 py-3 bg-white hover:bg-zinc-200 disabled:bg-zinc-700 text-zinc-900 disabled:text-zinc-500 font-medium rounded-full transition-colors flex items-center gap-2"
                            >
                                Send
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                </svg>
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
