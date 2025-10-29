"use client";
import { useEffect, useState } from "react";
import { getSession } from "@/lib/auth";
import { API_BASE } from "@/lib/api";
import Link from "next/link";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import KeyMetrics from "@/components/KeyMetrics";
import QuickActions from "@/components/QuickActions";
import LeadPipeline from "@/components/LeadPipeline";
import InsuranceProducts from "@/components/InsuranceProducts";
import LeadScoring from "@/components/LeadScoring";
import LeadManagement from "@/components/LeadManagement";

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

  if (loading) return <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-blue-50 to-white text-slate-900 font-medium">Loading...</div>;

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Key Metrics */}
        <section>
          <KeyMetrics />
        </section>

        {/* Quick Actions */}
        <section>
          <QuickActions />
        </section>

        {/* Main Grid: Pipeline, Products, Scoring */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <LeadPipeline />
          </div>
          <div className="lg:col-span-2">
            <InsuranceProducts />
          </div>
        </section>

        {/* Scoring and Activity */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <LeadScoring />
          </div>
          <div>
            <LeadManagement />
          </div>
        </section>

        {/* Recent Leads Section */}
        <section className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-slate-900">Recent Leads</h2>
            <Link href="/dashboard/leads" className="text-sm text-blue-700 hover:text-blue-800 underline font-medium">View All</Link>
          </div>
          <div className="text-sm text-slate-600 font-medium">Showing up to 10 recent items</div>
        </section>
      </div>
    </DashboardLayout>
  );
}

