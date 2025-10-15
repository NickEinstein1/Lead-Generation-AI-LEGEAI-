"use client";
import { useEffect, useState } from "react";
import { getSession, logout } from "@/lib/auth";
import { API_BASE } from "@/lib/api";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function DashboardPage() {
  const router = useRouter();
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const s = getSession();
    if (!s?.sessionId) {
      router.replace("/login");
      return;
    }
    (async () => {
      try {
        const [health, metrics, leads] = await Promise.all([
          fetch(`${API_BASE}/health`).then(r => r.json()),
          fetch(`${API_BASE}/pipeline/metrics`).then(r => r.json()).catch(() => ({ status: "unknown", metrics: {} })),
          fetch(`${API_BASE}/leads`).then(r => r.json()),
        ]);
        setStats({ health, metrics, leads });
      } finally {
        setLoading(false);
      }
    })();
  }, [router]);

  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;

  return (
    <div className="min-h-screen bg-neutral-50">
      <header className="flex items-center justify-between px-6 py-4 border-b bg-white">
        <div className="font-semibold">LEGEAI</div>
        <nav className="flex gap-4 text-sm">
          <Link href="/dashboard" className="text-neutral-700">Dashboard</Link>
          <Link href="/leads/new" className="text-neutral-700">New Lead</Link>
        </nav>
        <button onClick={() => { logout(); router.push("/login"); }} className="text-sm">Logout</button>
      </header>
      <main className="p-6 space-y-6">
        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white border rounded p-4"><div className="text-sm text-neutral-500">API</div><div className="text-2xl font-semibold">{stats?.health?.status || "?"}</div></div>
          <div className="bg-white border rounded p-4"><div className="text-sm text-neutral-500">Events Processed</div><div className="text-2xl font-semibold">{stats?.metrics?.metrics?.events_processed ?? 0}</div></div>
          <div className="bg-white border rounded p-4"><div className="text-sm text-neutral-500">Leads</div><div className="text-2xl font-semibold">{stats?.leads?.total ?? 0}</div></div>
        </section>
        <section className="bg-white border rounded p-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-semibold">Recent Leads</h2>
            <Link href="/leads/new" className="text-sm underline">Create lead</Link>
          </div>
          <div className="text-sm text-neutral-600">Showing up to 10 recent items</div>
        </section>
      </main>
    </div>
  );
}

