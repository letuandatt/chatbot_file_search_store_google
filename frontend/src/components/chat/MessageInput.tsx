"use client";

import type { ChangeEvent, KeyboardEvent } from "react";
import { formatFileName } from "@/lib/formatters";
import { isFileReady, type UploadedFile } from "@/types/chat";

interface MessageInputProps {
    inputMessage: string;
    onInputChange: (value: string) => void;
    onSend: () => void;
    isSending: boolean;
    uploadingFile: boolean;
    uploadedFile: UploadedFile | null;
    onClearUploadedFile: () => void;
    onFileUpload: (e: ChangeEvent<HTMLInputElement>) => void;
}

/**
 * The chat input row at the bottom: optional file pill (still
 * uploading / ready / error), the paperclip upload button, the
 * text field, and the send button. Disable rules are funnelled
 * through `isFileReady()` so this component is the single source
 * of truth for "can I send right now?".
 */
export function MessageInput({
    inputMessage,
    onInputChange,
    onSend,
    isSending,
    uploadingFile,
    uploadedFile,
    onClearUploadedFile,
    onFileUpload,
}: MessageInputProps) {
    const ready = isFileReady(uploadedFile);
    const sendDisabled = !inputMessage.trim() || isSending || !ready;

    const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && ready) onSend();
    };

    return (
        <div className="p-4 border-t border-zinc-800">
            <div className="max-w-4xl mx-auto flex items-center gap-3">
                {uploadedFile && (
                    <div
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs ${
                            uploadedFile.status === "processed" ||
                            uploadedFile.status === "ready"
                                ? "bg-emerald-500/20 text-emerald-400"
                                : uploadedFile.status === "error"
                                  ? "bg-red-500/20 text-red-400"
                                  : "bg-amber-500/20 text-amber-400"
                        }`}
                    >
                        {uploadedFile.status !== "processed" &&
                        uploadedFile.status !== "ready" &&
                        uploadedFile.status !== "error" ? (
                            <div className="animate-spin w-3 h-3 border border-current border-t-transparent rounded-full"></div>
                        ) : uploadedFile.status === "processed" ||
                          uploadedFile.status === "ready" ? (
                            <>
                                {uploadedFile.fileType === "image" ? (
                                    <svg
                                        className="w-3 h-3"
                                        fill="currentColor"
                                        viewBox="0 0 20 20"
                                    >
                                        <path
                                            fillRule="evenodd"
                                            d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"
                                            clipRule="evenodd"
                                        />
                                    </svg>
                                ) : (
                                    <svg
                                        className="w-3 h-3"
                                        fill="currentColor"
                                        viewBox="0 0 20 20"
                                    >
                                        <path
                                            fillRule="evenodd"
                                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                            clipRule="evenodd"
                                        />
                                    </svg>
                                )}
                            </>
                        ) : (
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                <path
                                    fillRule="evenodd"
                                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                                    clipRule="evenodd"
                                />
                            </svg>
                        )}
                        <span>{formatFileName(uploadedFile.name)}</span>
                        <button
                            onClick={onClearUploadedFile}
                            className="hover:text-white ml-1"
                            aria-label="Xóa file đã tải"
                        >
                            <svg
                                className="w-3 h-3"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M6 18L18 6M6 6l12 12"
                                />
                            </svg>
                        </button>
                    </div>
                )}

                <label className="p-3 hover:bg-zinc-800 rounded-xl cursor-pointer transition-colors">
                    <input
                        type="file"
                        accept=".pdf,image/*"
                        className="hidden"
                        onChange={onFileUpload}
                        disabled={uploadingFile || (uploadedFile !== null && !ready)}
                    />
                    {uploadingFile ? (
                        <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                    ) : (
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
                                d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                            />
                        </svg>
                    )}
                </label>

                <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => onInputChange(e.target.value)}
                    onKeyDown={handleKey}
                    placeholder={
                        uploadedFile !== null && !ready
                            ? "Đang xử lý file..."
                            : uploadedFile?.fileType === "image"
                              ? "Hỏi về ảnh này..."
                              : "Gõ câu hỏi..."
                    }
                    className="flex-1 h-12 px-4 bg-zinc-800/50 border border-zinc-700 rounded-xl text-white placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none transition-colors"
                    disabled={isSending || (uploadedFile !== null && !ready)}
                />

                <button
                    onClick={onSend}
                    disabled={sendDisabled}
                    className="px-5 py-3 bg-white hover:bg-zinc-200 disabled:bg-zinc-700 text-zinc-900 disabled:text-zinc-500 font-medium rounded-full transition-colors flex items-center gap-2"
                >
                    Send
                    <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M14 5l7 7m0 0l-7 7m7-7H3"
                        />
                    </svg>
                </button>
            </div>
        </div>
    );
}
