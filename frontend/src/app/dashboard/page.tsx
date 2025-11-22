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
  const [session, setSession] = useState<any>(null);

  useEffect(() => {
    const s = getSession();
    if (!s?.sessionId) {
      router.replace("/login");
      return;
    }
    setSession(s);
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
      <div className="space-y-4 sm:space-y-6">
        {/* Futuristic Welcome Banner */}
        <section className="relative bg-slate-900/60 backdrop-blur-xl rounded-3xl p-6 sm:p-8 text-white shadow-2xl overflow-hidden border border-cyan-500/30 hover:border-cyan-400/50 transition-all duration-500 group">
          {/* Animated gradient background */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600/30 via-orange-600/30 to-cyan-600/30 animate-gradient-shift"></div>

          {/* Glow orbs */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/20 rounded-full -mr-32 -mt-32 blur-3xl animate-pulse-slow"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-orange-500/20 rounded-full -ml-32 -mb-32 blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }}></div>
          <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-blue-500/10 rounded-full -ml-48 -mt-48 blur-3xl animate-pulse-slow" style={{ animationDelay: '2s' }}></div>

          {/* Grid overlay */}
          <div className="absolute inset-0 opacity-5" style={{
            backgroundImage: `
              linear-gradient(rgba(34, 211, 238, 0.5) 1px, transparent 1px),
              linear-gradient(90deg, rgba(34, 211, 238, 0.5) 1px, transparent 1px)
            `,
            backgroundSize: '30px 30px'
          }}></div>

          <div className="relative z-10">
            <h1 className="text-2xl sm:text-3xl lg:text-5xl font-bold mb-2 bg-gradient-to-r from-cyan-300 via-blue-300 to-orange-300 bg-clip-text text-transparent animate-fade-in">
              Welcome back, {session?.userId || "User"}! 👋
            </h1>
            <p className="text-cyan-200 text-sm sm:text-base lg:text-lg font-medium">Here's what's happening with your insurance leads today.</p>
          </div>

          {/* Corner accent */}
          <div className="absolute top-0 right-0 w-32 h-32 border-t-2 border-r-2 border-cyan-400/30 rounded-tr-3xl"></div>
          <div className="absolute bottom-0 left-0 w-32 h-32 border-b-2 border-l-2 border-cyan-400/30 rounded-bl-3xl"></div>
        </section>

        {/* Key Metrics */}
        <section>
          <KeyMetrics metrics={keyMetrics} />
        </section>

        {/* Quick Actions */}
        <section>
          <QuickActions />
        </section>

        {/* Main Grid: Pipeline, Products, Scoring - Responsive */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          <div className="md:col-span-2 lg:col-span-1">
            <LeadPipeline stages={pipelineStages} />
          </div>
          <div className="md:col-span-2 lg:col-span-2">
            <InsuranceProducts products={products} />
          </div>
        </section>

        {/* Scoring and Activity - Responsive */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
          <div>
            <LeadScoring metrics={scoringMetrics} overallScore={stats?.lead_scoring?.overall_score} />
          </div>
          <div>
            <LeadManagement activities={stats?.recent_activities} />
          </div>
        </section>

        {/* Recent Leads Section - Futuristic */}
        <section className="bg-slate-900/60 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-4 sm:p-6 shadow-2xl hover:shadow-cyan-500/20 transition-all duration-300 relative overflow-hidden">
          {/* Background glow */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-orange-500/10 to-cyan-500/10"></div>

          <div className="flex items-center justify-between mb-3 sm:mb-4 relative z-10">
            <h2 className="text-lg sm:text-xl font-bold text-blue-100 flex items-center gap-2">
              <span className="text-2xl">📋</span>
              <span className="bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">Recent Leads</span>
            </h2>
            <Link href="/dashboard/leads" className="text-xs sm:text-sm text-cyan-300 hover:text-cyan-200 font-bold hover:underline transition-all flex items-center gap-1 group">
              View All
              <span className="group-hover:translate-x-1 transition-transform">→</span>
            </Link>
          </div>
          <div className="text-xs sm:text-sm text-cyan-400 font-medium bg-cyan-500/20 backdrop-blur-sm px-3 py-2 rounded-lg inline-block border border-cyan-400/30 relative z-10">
            📊 Showing up to 10 recent items
          </div>
        </section>
      </div>
    </DashboardLayout>
  );
}

