"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { ChatHeader } from "@/components/chat/ChatHeader";
import { MessageInput } from "@/components/chat/MessageInput";
import { MessageList } from "@/components/chat/MessageList";
import { Sidebar } from "@/components/chat/Sidebar";
import {
    createNewSession,
    deleteAccountApi,
    deleteAllSessionsApi,
    deleteSessionApi,
    getFileStatus,
    getSessionMessages,
    listSessions,
    renameSessionApi,
    sendImageMessage,
    sendTextMessage,
    uploadPdf,
} from "@/lib/chatApi";
import type { Message, Session, UploadedFile } from "@/types/chat";

/**
 * Chat page composition root. Owns all state; delegates rendering
 * to <Sidebar>, <ChatHeader>, <MessageList>, <MessageInput>, and
 * dispatches HTTP work to lib/chatApi.
 *
 * The earlier version of this file was 844 lines that mixed fetch
 * code, state, polling, and JSX into a single component, which made
 * targeted changes risky. The slim version below keeps the same
 * behaviour but moves each concern into its own module so future
 * edits touch one file at a time.
 */
export default function ChatPage() {
    const router = useRouter();

    const [sessions, setSessions] = useState<Session[]>([]);
    const [activeSession, setActiveSession] = useState<string | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputMessage, setInputMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [uploadingFile, setUploadingFile] = useState(false);
    const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
    const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
    const [editingTitle, setEditingTitle] = useState("");
    const [expandedThinking, setExpandedThinking] = useState<Set<number>>(new Set());

    // ----- session bootstrap ------------------------------------------------

    const goToLogin = useCallback(() => router.push("/login"), [router]);

    const loadSessions = useCallback(async () => {
        const data = await listSessions({ onUnauthorized: goToLogin });
        if (data) setSessions(data);
    }, [goToLogin]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        if (!localStorage.getItem("access_token")) {
            router.push("/login");
            return;
        }
        loadSessions();
        // We intentionally only re-run when goToLogin changes (route changes
        // would require new auth). Adding loadSessions here would loop.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // ----- session CRUD -----------------------------------------------------

    const handleCreateSession = async () => {
        const newSession = await createNewSession({ onUnauthorized: goToLogin });
        if (!newSession) return;
        setSessions([newSession, ...sessions]);
        setActiveSession(newSession.session_id);
        setMessages([]);
    };

    const handleRenameSession = async (sessionId: string) => {
        const title = editingTitle.trim();
        if (!title) {
            setEditingSessionId(null);
            return;
        }
        const ok = await renameSessionApi(sessionId, title, { onUnauthorized: goToLogin });
        if (ok) {
            setSessions(
                sessions.map((s) => (s.session_id === sessionId ? { ...s, title } : s)),
            );
        }
        setEditingSessionId(null);
    };

    const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm("Xóa phiên làm việc này?")) return;
        const ok = await deleteSessionApi(sessionId, { onUnauthorized: goToLogin });
        if (!ok) return;
        setSessions(sessions.filter((s) => s.session_id !== sessionId));
        if (activeSession === sessionId) {
            setActiveSession(null);
            setMessages([]);
        }
    };

    const handleDeleteAllSessions = async () => {
        if (!confirm("Bạn có chắc muốn xóa tất cả phiên làm việc?")) return;
        const ok = await deleteAllSessionsApi({ onUnauthorized: goToLogin });
        if (!ok) return;
        setSessions([]);
        setActiveSession(null);
        setMessages([]);
    };

    const handleSelectSession = async (sessionId: string) => {
        setIsLoading(true);
        const data = await getSessionMessages(sessionId, { onUnauthorized: goToLogin });
        if (data) {
            setMessages(
                (data.messages || []).map((m: Message) => ({
                    ...m,
                    thinking_steps: m.thinking_steps || undefined,
                })),
            );
            setActiveSession(sessionId);
        }
        setIsLoading(false);
    };

    // ----- send message -----------------------------------------------------

    const sendMessage = async () => {
        if (!inputMessage.trim() || isSending) return;
        const userMessage = inputMessage.trim();
        setInputMessage("");
        setIsSending(true);

        setMessages((prev) => [
            ...prev,
            {
                question: userMessage,
                answer: "Đang xử lý...",
                timestamp: new Date().toISOString(),
            },
        ]);

        try {
            const data =
                uploadedFile?.fileType === "image" && uploadedFile.rawFile
                    ? await sendImageMessage(userMessage, uploadedFile.rawFile, activeSession)
                    : await sendTextMessage(userMessage, activeSession, {
                          onUnauthorized: goToLogin,
                      });

            if (uploadedFile?.fileType === "image") setUploadedFile(null);

            if (!data) throw new Error("Failed to send message");

            setMessages((prev) => {
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
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1].answer =
                    "Lỗi khi gửi tin nhắn. Vui lòng thử lại.";
                return updated;
            });
        } finally {
            setIsSending(false);
        }
    };

    // ----- file upload ------------------------------------------------------

    const pollFileStatus = (fileId: string, fileName: string) => {
        const MAX_ATTEMPTS = 60;
        let attempts = 0;
        const tick = async () => {
            const data = await getFileStatus(fileId, { onUnauthorized: goToLogin });
            if (data?.status === "processed") {
                setUploadedFile({
                    name: fileName,
                    status: "processed",
                    fileType: "pdf",
                    fileId,
                });
                return;
            }
            if (data?.status === "error_processing") {
                setUploadedFile({
                    name: fileName,
                    status: "error",
                    fileType: "pdf",
                    fileId,
                });
                alert("File processing failed.");
                return;
            }
            attempts += 1;
            if (attempts < MAX_ATTEMPTS) {
                setTimeout(tick, 2000);
            } else {
                alert("File processing timeout.");
                setUploadedFile(null);
            }
        };
        setTimeout(tick, 1000);
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const isImage = file.type.startsWith("image/");
        const isPdf = file.name.toLowerCase().endsWith(".pdf");
        if (!isImage && !isPdf) {
            alert("Chỉ hỗ trợ file ảnh (jpg, png, gif, webp) hoặc PDF.");
            return;
        }

        if (isImage) {
            setUploadedFile({
                name: file.name,
                status: "ready",
                fileType: "image",
                rawFile: file,
            });
            e.target.value = "";
            return;
        }

        // PDF flow.
        setUploadingFile(true);
        setUploadedFile({ name: file.name, status: "uploading", fileType: "pdf" });
        try {
            const data = await uploadPdf(file, activeSession);
            if (!data) {
                alert("Lỗi khi tải file lên.");
                setUploadedFile(null);
                return;
            }
            setUploadedFile({
                name: file.name,
                status: "uploaded",
                fileType: "pdf",
                fileId: data.file_id,
            });
            pollFileStatus(data.file_id, file.name);
        } finally {
            setUploadingFile(false);
            e.target.value = "";
        }
    };

    // ----- auth actions -----------------------------------------------------

    const handleLogout = () => {
        if (!confirm("Are u sure you want to logout?")) return;
        localStorage.removeItem("access_token");
        router.push("/login");
    };

    const handleDeleteAccount = async () => {
        if (
            !confirm("Bạn có chắc chắn muốn XÓA TÀI KHOẢN? Hành động này không thể hoàn tác!")
        )
            return;
        const ok = await deleteAccountApi({ onUnauthorized: goToLogin });
        if (!ok) {
            alert("Lỗi khi xóa tài khoản.");
            return;
        }
        localStorage.removeItem("access_token");
        alert("Tài khoản đã được xóa thành công.");
        router.push("/");
    };

    // ----- render -----------------------------------------------------------

    const toggleThinking = (idx: number) => {
        setExpandedThinking((prev) => {
            const next = new Set(prev);
            if (next.has(idx)) next.delete(idx);
            else next.add(idx);
            return next;
        });
    };

    return (
        <div className="h-screen flex bg-zinc-950 text-white">
            <Sidebar
                collapsed={sidebarCollapsed}
                sessions={sessions}
                activeSession={activeSession}
                editingSessionId={editingSessionId}
                editingTitle={editingTitle}
                onCreateSession={handleCreateSession}
                onDeleteAllSessions={handleDeleteAllSessions}
                onSelectSession={handleSelectSession}
                onStartRename={(id, currentTitle) => {
                    setEditingSessionId(id);
                    setEditingTitle(currentTitle);
                }}
                onCommitRename={handleRenameSession}
                onCancelRename={() => setEditingSessionId(null)}
                onChangeEditingTitle={setEditingTitle}
                onDeleteSession={handleDeleteSession}
                onOpenProfile={() => router.push("/profile")}
                onLogout={handleLogout}
                onDeleteAccount={handleDeleteAccount}
            />

            <div className="flex-1 flex flex-col">
                <ChatHeader
                    onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
                    onResetSession={() => {
                        setActiveSession(null);
                        setMessages([]);
                    }}
                    onOpenFiles={() => router.push("/files")}
                />

                <div className="flex-1 overflow-y-auto p-6">
                    <MessageList
                        messages={messages}
                        isLoading={isLoading}
                        hasActiveSession={!!activeSession}
                        expandedThinking={expandedThinking}
                        onToggleThinking={toggleThinking}
                    />
                </div>

                {activeSession && (
                    <MessageInput
                        inputMessage={inputMessage}
                        onInputChange={setInputMessage}
                        onSend={sendMessage}
                        isSending={isSending}
                        uploadingFile={uploadingFile}
                        uploadedFile={uploadedFile}
                        onClearUploadedFile={() => setUploadedFile(null)}
                        onFileUpload={handleFileUpload}
                    />
                )}
            </div>
        </div>
    );
}
