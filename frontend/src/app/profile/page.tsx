"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface UserProfile {
    id: string;
    email: string;
    full_name: string | null;
    avatar_url: string | null;
    created_at: string;
    is_active: boolean;
}

export default function ProfilePage() {
    const router = useRouter();
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [editingName, setEditingName] = useState(false);
    const [newName, setNewName] = useState("");
    const [savingName, setSavingName] = useState(false);

    // Password change
    const [currentPassword, setCurrentPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [changingPassword, setChangingPassword] = useState(false);
    const [passwordError, setPasswordError] = useState("");

    // Auth is now driven by httpOnly cookies set on login; the frontend
    // never reads the token itself. Every fetch must include credentials.
    const jsonHeaders = { "Content-Type": "application/json" } as const;

    // Load profile
    useEffect(() => {
        loadProfile();
    }, []);

    const loadProfile = async () => {
        try {
            const res = await fetch(`${API_URL}/users/me`, {
                credentials: "include",
            });
            if (res.ok) {
                const data = await res.json();
                setProfile(data);
                setNewName(data.full_name || "");
            } else if (res.status === 401) {
                router.push("/login");
            }
        } catch (error) {
            console.error("Error loading profile:", error);
        } finally {
            setIsLoading(false);
        }
    };

    // Update name
    const handleSaveName = async () => {
        if (!newName.trim()) return;
        setSavingName(true);
        try {
            const res = await fetch(`${API_URL}/users/me`, {
                method: "PUT",
                credentials: "include",
                headers: jsonHeaders,
                body: JSON.stringify({ full_name: newName.trim() }),
            });
            if (res.ok) {
                const data = await res.json();
                setProfile(data);
                setEditingName(false);
            } else {
                alert("Lỗi khi cập nhật tên.");
            }
        } catch (error) {
            console.error("Error updating name:", error);
            alert("Lỗi khi cập nhật tên.");
        } finally {
            setSavingName(false);
        }
    };

    // Change password
    const handleChangePassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setPasswordError("");

        if (newPassword.length < 6) {
            setPasswordError("Mật khẩu mới phải có ít nhất 6 ký tự.");
            return;
        }
        if (newPassword !== confirmPassword) {
            setPasswordError("Mật khẩu xác nhận không khớp.");
            return;
        }

        setChangingPassword(true);
        try {
            const res = await fetch(`${API_URL}/users/me/change-password`, {
                method: "POST",
                credentials: "include",
                headers: jsonHeaders,
                body: JSON.stringify({
                    current_password: currentPassword,
                    new_password: newPassword,
                }),
            });
            if (res.ok) {
                alert("Đổi mật khẩu thành công!");
                setCurrentPassword("");
                setNewPassword("");
                setConfirmPassword("");
            } else {
                const data = await res.json();
                setPasswordError(data.detail || "Mật khẩu hiện tại không đúng.");
            }
        } catch (error) {
            console.error("Error changing password:", error);
            setPasswordError("Lỗi khi đổi mật khẩu.");
        } finally {
            setChangingPassword(false);
        }
    };

    // Format date
    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return date.toLocaleString("vi-VN", {
            hour: "2-digit",
            minute: "2-digit",
            day: "2-digit",
            month: "2-digit",
            year: "numeric",
        });
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
            <div className="max-w-lg mx-auto px-4 py-8">
                {/* Back button */}
                <Link
                    href="/chat"
                    className="inline-flex items-center gap-2 px-4 py-2 border border-zinc-700 rounded-full text-zinc-400 hover:text-white hover:border-zinc-500 transition-colors mb-8"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                    </svg>
                    Quay lại
                </Link>

                {/* Logo placeholder */}
                <div className="flex justify-center mb-6">
                    <div className="w-12 h-12 rounded-xl bg-white flex items-center justify-center">
                        <svg className="w-7 h-7 text-zinc-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                    </div>
                </div>

                {/* Title */}
                <h1 className="text-2xl font-bold text-center mb-8">Thông tin tài khoản</h1>

                {/* Account Info */}
                <div className="space-y-4 mb-10">
                    {/* Name */}
                    <div className="flex items-center justify-between py-3 border-b border-zinc-800">
                        <span className="text-zinc-500">Tên:</span>
                        <div className="flex items-center gap-2">
                            {editingName ? (
                                <>
                                    <input
                                        type="text"
                                        value={newName}
                                        onChange={(e) => setNewName(e.target.value)}
                                        className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1 text-sm text-white focus:outline-none focus:border-zinc-500"
                                        onKeyDown={(e) => {
                                            if (e.key === "Enter") handleSaveName();
                                            if (e.key === "Escape") {
                                                setEditingName(false);
                                                setNewName(profile?.full_name || "");
                                            }
                                        }}
                                        autoFocus
                                    />
                                    <button
                                        onClick={handleSaveName}
                                        disabled={savingName}
                                        className="text-emerald-400 hover:text-emerald-300"
                                    >
                                        {savingName ? (
                                            <div className="animate-spin w-4 h-4 border border-current border-t-transparent rounded-full"></div>
                                        ) : (
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                            </svg>
                                        )}
                                    </button>
                                    <button
                                        onClick={() => {
                                            setEditingName(false);
                                            setNewName(profile?.full_name || "");
                                        }}
                                        className="text-zinc-500 hover:text-zinc-300"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                        </svg>
                                    </button>
                                </>
                            ) : (
                                <>
                                    <span className="text-blue-400">{profile?.full_name || "Chưa đặt tên"}</span>
                                    <button
                                        onClick={() => setEditingName(true)}
                                        className="text-zinc-500 hover:text-zinc-300"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                                        </svg>
                                    </button>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Email */}
                    <div className="flex items-center justify-between py-3 border-b border-zinc-800">
                        <span className="text-zinc-500">Email:</span>
                        <span className="text-zinc-300">{profile?.email}</span>
                    </div>

                    {/* Created at */}
                    <div className="flex items-center justify-between py-3 border-b border-zinc-800">
                        <span className="text-zinc-500">Ngày tạo:</span>
                        <span className="text-zinc-300">{profile ? formatDate(profile.created_at) : ""}</span>
                    </div>
                </div>

                {/* Change Password Section */}
                <h2 className="text-xl font-semibold text-center mb-6">Đổi mật khẩu</h2>

                <form onSubmit={handleChangePassword} className="space-y-4">
                    <div>
                        <label className="block text-sm text-zinc-500 mb-2">Mật khẩu hiện tại:</label>
                        <input
                            type="password"
                            value={currentPassword}
                            onChange={(e) => setCurrentPassword(e.target.value)}
                            className="w-full h-12 px-4 bg-zinc-900 border border-zinc-800 rounded-xl text-white placeholder:text-zinc-600 focus:border-zinc-600 focus:outline-none transition-colors"
                            required
                        />
                    </div>

                    <div>
                        <label className="block text-sm text-zinc-500 mb-2">Mật khẩu mới (ít nhất 6 ký tự):</label>
                        <input
                            type="password"
                            value={newPassword}
                            onChange={(e) => setNewPassword(e.target.value)}
                            className="w-full h-12 px-4 bg-zinc-900 border border-zinc-800 rounded-xl text-white placeholder:text-zinc-600 focus:border-zinc-600 focus:outline-none transition-colors"
                            minLength={6}
                            required
                        />
                    </div>

                    <div>
                        <label className="block text-sm text-zinc-500 mb-2">Xác nhận mật khẩu mới:</label>
                        <input
                            type="password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            className="w-full h-12 px-4 bg-zinc-900 border border-zinc-800 rounded-xl text-white placeholder:text-zinc-600 focus:border-zinc-600 focus:outline-none transition-colors"
                            required
                        />
                    </div>

                    {passwordError && (
                        <p className="text-red-400 text-sm text-center">{passwordError}</p>
                    )}

                    <button
                        type="submit"
                        disabled={changingPassword}
                        className="w-full h-12 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 text-white font-medium rounded-xl transition-colors flex items-center justify-center gap-2"
                    >
                        {changingPassword ? (
                            <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                        ) : (
                            "Đổi mật khẩu"
                        )}
                    </button>
                </form>
            </div>
        </div>
    );
}
