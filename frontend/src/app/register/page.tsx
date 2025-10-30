"use client";
import { useState } from "react";
import { register, setSession } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function RegisterPage() {
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setLoading(true);
    const fd = new FormData(e.currentTarget);
    const email = String(fd.get("email") || "");
    const username = String(fd.get("username") || email);
    const password = String(fd.get("password") || "");
    try {
      const res = await register({ username, email, password });
      setSession({ sessionId: res.session_id, token: res.token, userId: res.user_id, role: "agent" });
      router.push("/dashboard");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Registration failed");
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
        <h1 className="text-2xl font-semibold text-neutral-900">Create account</h1>
        <input name="email" type="email" placeholder="Email" className="w-full border border-border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-primary/60" />
        <input name="username" placeholder="Username (optional)" className="w-full border border-border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-primary/60" />
        <input name="password" type="password" placeholder="Password" className="w-full border border-border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-primary/60" />
        {error && <div className="text-sm text-red-600">{error}</div>}
        <button disabled={loading} className="w-full bg-primary text-primary-foreground py-2.5 rounded-md shadow-sm hover:shadow transition">{loading ? "Creating..." : "Create account"}</button>
        <div className="text-sm text-neutral-700">Have an account? <a href="/login" className="underline">Sign in</a></div>
      </form>
    </div>
  );
}

