"use client";
import { useState } from "react";
import { login, setSession } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function LoginPage() {
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setLoading(true);
    const fd = new FormData(e.currentTarget);
    const username = String(fd.get("username") || "");
    const password = String(fd.get("password") || "");
    try {
      const res = await login({ username, password });
      setSession({ sessionId: res.session_id, token: res.token, userId: res.user_id, role: res.role });
      router.push("/dashboard");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-neutral-50 via-white to-blue-50 relative">
      <Link href="/" className="absolute top-4 left-4 text-sm text-neutral-800 hover:text-blue-600">← Back to home</Link>
      <form onSubmit={onSubmit} className="w-full max-w-sm bg-white border border-neutral-200 rounded-xl p-6 space-y-4 shadow-lg">
        <h1 className="text-2xl font-semibold text-neutral-900">Sign in</h1>
        <div>
          <input
            name="username"
            placeholder="Email or username"
            className="w-full border border-neutral-300 px-3 py-2 rounded text-neutral-900 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <div>
          <input
            name="password"
            type="password"
            placeholder="Password"
            className="w-full border border-neutral-300 px-3 py-2 rounded text-neutral-900 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        {error && <div className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</div>}
        <button
          disabled={loading}
          className="w-full bg-blue-600 text-white font-medium py-2.5 rounded-md shadow-sm hover:bg-blue-700 transition disabled:opacity-50"
        >
          {loading ? "Signing in..." : "Sign in"}
        </button>
        <div className="text-sm text-neutral-700">No account? <a href="/register" className="text-blue-600 hover:text-blue-700 underline font-medium">Create one</a></div>
      </form>
    </div>
  );
}

