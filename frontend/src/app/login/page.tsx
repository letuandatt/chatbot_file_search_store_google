"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function LoginPage() {
    const router = useRouter();
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState("");
    const [formData, setFormData] = useState({ email: "", password: "" });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
        setError("");
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError("");

        try {
            // `credentials: "include"` lets the browser store the httpOnly
            // access_token / refresh_token cookies set by the backend.
            const res = await fetch(`${API_URL}/auth/login`, {
                method: "POST",
                credentials: "include",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Đăng nhập thất bại");
            router.push("/chat");
        } catch (err) {
            setError(err instanceof Error ? err.message : "Đã xảy ra lỗi");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10 bg-zinc-950 text-white">
            {/* Logo */}
            <Link href="/" className="w-14 h-14 rounded-2xl bg-white flex items-center justify-center hover:scale-105 transition-transform">
                <svg className="w-7 h-7 text-zinc-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
            </Link>

            {/* Title */}
            <h1 className="text-3xl font-bold mt-6">Đăng nhập</h1>
            <p className="text-zinc-500 mt-2">Chào mừng trở lại!</p>

            {/* Form */}
            <form onSubmit={handleSubmit} className="w-full max-w-sm mt-8">
                {error && (
                    <div className="text-red-400 text-sm text-center py-2 mb-4">
                        {error}
                    </div>
                )}

                <div className="mb-5">
                    <label className="block text-sm text-zinc-400 mb-2">Email</label>
                    <input
                        name="email"
                        type="email"
                        required
                        value={formData.email}
                        onChange={handleChange}
                        placeholder="email@example.com"
                        className="w-full h-12 px-4 rounded-xl border border-zinc-800 bg-transparent text-white placeholder:text-zinc-600 focus:border-zinc-600 focus:outline-none transition-colors"
                    />
                </div>

                <div className="mb-6">
                    <label className="block text-sm text-zinc-400 mb-2">Mật khẩu</label>
                    <input
                        name="password"
                        type="password"
                        required
                        value={formData.password}
                        onChange={handleChange}
                        placeholder="••••••••"
                        className="w-full h-12 px-4 rounded-xl border border-zinc-800 bg-transparent text-white placeholder:text-zinc-600 focus:border-zinc-600 focus:outline-none transition-colors"
                    />
                </div>

                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full h-12 rounded-xl bg-white text-zinc-900 font-medium hover:bg-zinc-200 disabled:opacity-50 transition-colors"
                >
                    {isLoading ? "Đang xử lý..." : "Đăng nhập"}
                </button>
            </form>

            {/* Footer */}
            <p className="text-zinc-500 text-sm mt-8">
                Chưa có tài khoản?{" "}
                <Link href="/register" className="text-white hover:underline">
                    Đăng ký ngay
                </Link>
            </p>

            <p className="text-zinc-700 text-xs mt-6">© 2025 Law Chatbot. All rights reserved.</p>
        </div>
    );
}
