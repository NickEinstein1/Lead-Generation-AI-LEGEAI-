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
        const dashboardStats = await fetch(`${API_BASE}/dashboard/stats`)
          .then(r => r.json())
          .catch(() => null);

        setStats(dashboardStats);
      } finally {
        setLoading(false);
      }
    })();
  }, [router]);

  if (loading) return <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-blue-50 to-white text-slate-900 font-medium">Loading...</div>;

  // Prepare data for components from API response
  const keyMetrics = stats?.key_metrics ? [
    {
      label: "Total Leads",
      value: stats.key_metrics.total_leads,
      change: stats.key_metrics.total_leads_change,
      icon: "👥",
      color: "from-blue-500 to-blue-600",
    },
    {
      label: "This Month",
      value: stats.key_metrics.monthly_leads,
      change: stats.key_metrics.monthly_leads_change,
      icon: "📈",
      color: "from-green-500 to-green-600",
    },
    {
      label: "Conversion Rate",
      value: `${stats.key_metrics.conversion_rate}%`,
      change: stats.key_metrics.conversion_rate_change,
      icon: "🎯",
      color: "from-purple-500 to-purple-600",
    },
    {
      label: "Avg Deal Value",
      value: `$${stats.key_metrics.avg_deal_value.toLocaleString()}`,
      change: stats.key_metrics.avg_deal_value_change,
      icon: "💰",
      color: "from-amber-500 to-amber-600",
    },
  ] : undefined;

  const pipelineStages = stats?.pipeline?.stages?.map((stage: any) => ({
    name: stage.name,
    count: stage.count,
    percentage: stage.percentage,
    color: stage.name === "New" ? "bg-blue-500" :
           stage.name === "Contacted" ? "bg-cyan-500" :
           stage.name === "Qualified" ? "bg-emerald-500" :
           stage.name === "Proposal" ? "bg-amber-500" : "bg-green-600"
  }));

  const scoringMetrics = stats?.lead_scoring?.metrics?.map((m: any) => ({
    label: m.label,
    value: m.value,
    max: m.max,
    color: m.label === "Engagement" ? "bg-blue-500" :
           m.label === "Budget Fit" ? "bg-green-500" :
           m.label === "Timeline" ? "bg-amber-500" :
           m.label === "Authority" ? "bg-purple-500" : "bg-red-500"
  }));

  const products = stats?.products?.map((p: any) => ({
    id: p.id,
    name: p.name,
    icon: p.id === "auto" ? "🚗" :
          p.id === "home" ? "🏠" :
          p.id === "life" ? "❤️" : "⚕️",
    leads: p.leads,
    revenue: `$${p.revenue.toLocaleString()}`,
    conversionRate: p.conversion_rate,
    color: p.id === "auto" ? "from-blue-500 to-blue-600" :
           p.id === "home" ? "from-amber-500 to-amber-600" :
           p.id === "life" ? "from-red-500 to-red-600" : "from-green-500 to-green-600"
  }));

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Key Metrics */}
        <section>
          <KeyMetrics metrics={keyMetrics} />
        </section>

        {/* Quick Actions */}
        <section>
          <QuickActions />
        </section>

        {/* Main Grid: Pipeline, Products, Scoring */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <LeadPipeline stages={pipelineStages} />
          </div>
          <div className="lg:col-span-2">
            <InsuranceProducts products={products} />
          </div>
        </section>

        {/* Scoring and Activity */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <LeadScoring metrics={scoringMetrics} overallScore={stats?.lead_scoring?.overall_score} />
          </div>
          <div>
            <LeadManagement activities={stats?.recent_activities} />
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

