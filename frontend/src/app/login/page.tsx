"use client";
import { useState } from "react";
import { login, setSession } from "@/lib/auth";
import { useRouter } from "next/navigation";

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
    <div className="min-h-screen flex items-center justify-center bg-neutral-50">
      <form onSubmit={onSubmit} className="w-full max-w-sm bg-white border rounded p-6 space-y-4 shadow-sm">
        <h1 className="text-xl font-semibold text-neutral-900">Sign in</h1>
        <input name="username" placeholder="Email or username" className="w-full border px-3 py-2 rounded" />
        <input name="password" type="password" placeholder="Password" className="w-full border px-3 py-2 rounded" />
        {error && <div className="text-sm text-red-600">{error}</div>}
        <button disabled={loading} className="w-full bg-black text-white py-2 rounded">{loading ? "Signing in..." : "Sign in"}</button>
        <div className="text-sm text-neutral-600">No account? <a href="/register" className="underline">Create one</a></div>
      </form>
    </div>
  );
}

