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
    <div className="relative min-h-screen flex items-center justify-center">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,0,0,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,0,0,0.03)_1px,transparent_1px)] bg-[size:24px_24px]" />
        <div className="absolute -top-32 right-[-10%] h-[420px] w-[420px] opacity-40 blur-3xl rounded-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-300/20 via-blue-400/10 to-transparent" />
      </div>
      <Link href="/" className="absolute top-4 left-4 text-sm text-neutral-900 hover:text-primary">← Back to home</Link>
      <form onSubmit={onSubmit} className="w-full max-w-sm bg-white/85 backdrop-blur border rounded-xl p-6 space-y-4 shadow-md">
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

