"use client";
import Link from "next/link";

export default function Landing() {
  return (
    <div className="min-h-screen bg-neutral-50 flex flex-col">
      <header className="flex items-center justify-between px-6 py-4">
        <div className="font-semibold">LEGEAI</div>
        <nav className="flex gap-4 text-sm">
          <Link href="/login" className="text-neutral-700">Login</Link>
          <Link href="/register" className="text-neutral-700">Register</Link>
        </nav>
      </header>
      <main className="flex-1 flex items-center justify-center">
        <div className="text-center px-6">
          <h1 className="text-3xl md:text-5xl font-semibold text-neutral-900">Lead Generation CRM for Insurance</h1>
          <p className="mt-4 text-neutral-600 max-w-xl mx-auto">Minimal, fast, and compliant lead intake, scoring, routing and dashboards. Built for real-time operations.</p>
          <div className="mt-6 flex items-center justify-center gap-3">
            <Link href="/dashboard" className="bg-black text-white px-4 py-2 rounded">Go to app</Link>
            <Link href="/leads/new" className="border px-4 py-2 rounded">Create a lead</Link>
          </div>
        </div>
      </main>
      <footer className="px-6 py-4 text-xs text-neutral-500">Made with care. No purple, just clarity.</footer>
    </div>
  );
}
