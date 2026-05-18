"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface UploadedFile {
    file_id: string;
    filename: string;
    status: string;
    session_id: string;
    session_name: string;
    created_at: string;
    has_file: boolean;
}

export default function FilesPage() {
    const router = useRouter();
    const [files, setFiles] = useState<UploadedFile[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    // Auth is now driven by httpOnly cookies; the frontend never reads
    // the token directly. Every fetch must pass `credentials: "include"`.
    useEffect(() => {
        loadFiles();
    }, []);

    const loadFiles = async () => {
        try {
            const res = await fetch(`${API_URL}/chat/files`, {
                credentials: "include",
            });
            if (res.ok) {
                const data = await res.json();
                setFiles(data.files || []);
            } else if (res.status === 401) {
                router.push("/login");
            }
        } catch (error) {
            console.error("Error loading files:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleDownload = async (file: UploadedFile) => {
        if (!file.has_file) {
            alert("File gốc không còn khả dụng.");
            return;
        }

        try {
            const res = await fetch(`${API_URL}/chat/files/${file.file_id}/download`, {
                credentials: "include",
            });

            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = file.filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                alert("Lỗi khi tải file.");
            }
        } catch (error) {
            console.error("Error downloading file:", error);
            alert("Lỗi khi tải file.");
        }
    };

    const handleDelete = async (file: UploadedFile) => {
        if (!confirm(`Bạn có chắc muốn xóa file "${file.filename}"?`)) return;

        try {
            const res = await fetch(`${API_URL}/chat/files/${file.file_id}`, {
                method: "DELETE",
                credentials: "include",
            });

            if (res.ok) {
                setFiles(files.filter(f => f.file_id !== file.file_id));
            } else {
                alert("Lỗi khi xóa file.");
            }
        } catch (error) {
            console.error("Error deleting file:", error);
            alert("Lỗi khi xóa file.");
        }
    };

    // Format date
    const formatDate = (dateStr: string) => {
        if (!dateStr) return "N/A";
        const date = new Date(dateStr);
        return date.toLocaleString("vi-VN", {
            hour: "2-digit",
            minute: "2-digit",
            day: "2-digit",
            month: "2-digit",
            year: "numeric",
        });
    };

    // Truncate filename
    const formatFilename = (name: string, maxLen: number = 30) => {
        if (name.length <= maxLen) return name;
        const ext = name.split('.').pop() || '';
        const baseName = name.slice(0, name.lastIndexOf('.'));
        const truncatedBase = baseName.slice(0, maxLen - ext.length - 4);
        return `${truncatedBase}...${ext}`;
    };

    // Status badge
    const getStatusBadge = (status: string) => {
        const statusMap: Record<string, { color: string, text: string }> = {
            "processed": { color: "bg-emerald-500/20 text-emerald-400", text: "PROCESSED" },
            "uploaded": { color: "bg-amber-500/20 text-amber-400", text: "UPLOADED" },
            "error_processing": { color: "bg-red-500/20 text-red-400", text: "ERROR" },
        };
        const s = statusMap[status] || { color: "bg-zinc-500/20 text-zinc-400", text: status.toUpperCase() };
        return (
            <span className={`px-2 py-1 rounded text-xs font-medium ${s.color}`}>
                {s.text}
            </span>
        );
    };

    if (isLoading) {
        return (
            <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
                <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-zinc-950 text-white">
            <div className="max-w-4xl mx-auto px-4 py-8">
                {/* Back button */}
                <Link
                    href="/chat"
                    className="inline-flex items-center gap-2 px-4 py-2 border border-zinc-700 rounded-full text-zinc-400 hover:text-white hover:border-zinc-500 transition-colors mb-8"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                    </svg>
                    Quay lại Chat
                </Link>

                {/* Title */}
                <h1 className="text-2xl font-bold text-center mb-8">File đã tải lên</h1>

                {/* Files Table */}
                {files.length > 0 ? (
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-zinc-800 text-left">
                                    <th className="py-3 px-4 text-zinc-400 font-medium text-sm">Tên File</th>
                                    <th className="py-3 px-4 text-zinc-400 font-medium text-sm">Ngày tải lên</th>
                                    <th className="py-3 px-4 text-zinc-400 font-medium text-sm">Trạng thái</th>
                                    <th className="py-3 px-4 text-zinc-400 font-medium text-sm">Phiên</th>
                                    <th className="py-3 px-4 text-zinc-400 font-medium text-sm"></th>
                                </tr>
                            </thead>
                            <tbody>
                                {files.map((file) => (
                                    <tr key={file.file_id} className="border-b border-zinc-800/50 hover:bg-zinc-900/50 transition-colors">
                                        <td className="py-4 px-4">
                                            <span className="text-zinc-200" title={file.filename}>
                                                {formatFilename(file.filename)}
                                            </span>
                                        </td>
                                        <td className="py-4 px-4 text-zinc-400 text-sm">
                                            {formatDate(file.created_at)}
                                        </td>
                                        <td className="py-4 px-4">
                                            {getStatusBadge(file.status)}
                                        </td>
                                        <td className="py-4 px-4 text-zinc-400 text-sm" title={file.session_id}>
                                            {file.session_name.length > 20
                                                ? file.session_name.slice(0, 20) + "..."
                                                : file.session_name}
                                        </td>
                                        <td className="py-4 px-4">
                                            <div className="flex items-center gap-1">
                                                {file.has_file && (
                                                    <button
                                                        onClick={() => handleDownload(file)}
                                                        className="p-2 hover:bg-zinc-800 rounded-lg transition-colors text-blue-400 hover:text-blue-300"
                                                        title="Tải xuống"
                                                    >
                                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                                        </svg>
                                                    </button>
                                                )}
                                                <button
                                                    onClick={() => handleDelete(file)}
                                                    className="p-2 hover:bg-red-500/20 rounded-lg transition-colors text-zinc-500 hover:text-red-400"
                                                    title="Xóa file"
                                                >
                                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                    </svg>
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="text-center py-16">
                        <div className="w-16 h-16 rounded-xl bg-zinc-800 flex items-center justify-center mx-auto mb-4">
                            <svg className="w-8 h-8 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                            </svg>
                        </div>
                        <p className="text-zinc-500">Bạn chưa tải lên file nào.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
