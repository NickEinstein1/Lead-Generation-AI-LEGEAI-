"use client";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { getDashboardOverview, getLeadTimeseries, getScoreTimeseries } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import ConversionFunnel from "@/components/ConversionFunnel";
import KeyMetrics from "@/components/KeyMetrics";
import DashboardLayout from "@/components/DashboardLayout";

const MiniAccent3D = dynamic(() => import("@/components/Hero3D").then(m => m.default), { ssr: false });

export default function AnalyticsPage() {
  const [data, setData] = useState<{ date: string; leads: number; score: number }[]>([]);
  const [overview, setOverview] = useState<any>(null);

  useEffect(() => {
    (async () => {
      try {
        const [ovr, leadsTs, scoresTs] = await Promise.all([
          getDashboardOverview(),
          getLeadTimeseries(14),
          getScoreTimeseries(14),
        ]);
        setOverview(ovr?.overview || null);
        // Merge by date
        const leadsMap = new Map<string, number>(
          (leadsTs?.series || []).map((d: any) => [d.date, d.leads])
        );
        const scoreMap = new Map<string, number>(
          (scoresTs?.series || []).map((d: any) => [d.date, d.avg_score])
        );
        const dates = Array.from(new Set<string>([...leadsMap.keys(), ...scoreMap.keys()])).sort();
        setData(
          dates.map((d) => ({ date: d, leads: leadsMap.get(d) || 0, score: scoreMap.get(d) || 0 }))
        );
      } catch (e) {
        // fallback demo if API not ready
        setData([
          { date: "Mon", leads: 12, score: 62 },
          { date: "Tue", leads: 18, score: 68 },
          { date: "Wed", leads: 15, score: 64 },
          { date: "Thu", leads: 22, score: 72 },
          { date: "Fri", leads: 30, score: 78 },
          { date: "Sat", leads: 20, score: 69 },
          { date: "Sun", leads: 14, score: 66 },
        ]);
      }
    })();
  }, []);

  return (
    <DashboardLayout>
      <div className="relative overflow-hidden">
        <div className="absolute right-0 top-0 h-40 w-40 opacity-30 blur-2xl rounded-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-400/40 via-blue-500/20 to-transparent" />
        <div className="mx-auto max-w-6xl">
          <div className="flex items-center justify-between gap-6">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Analytics</h1>
              <p className="text-slate-700 font-medium">Funnel, sources, and model performance at a glance.</p>
            </div>
            <div className="h-16 w-32 opacity-70">
              <MiniAccent3D />
            </div>
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded border-2 border-blue-200 bg-white p-4 shadow-md hover:shadow-lg transition">
              <div className="text-slate-600 text-sm font-medium">Assignments (total)</div>
              <div className="text-2xl font-bold text-blue-700">{overview?.routing?.total_assignments ?? 0}</div>
            </div>
            <div className="rounded border-2 border-blue-200 bg-white p-4 shadow-md hover:shadow-lg transition">
              <div className="text-slate-600 text-sm font-medium">Avg Assignments/Rep</div>
              <div className="text-2xl font-bold text-blue-700">{overview?.routing?.average_assignments ?? 0}</div>
            </div>
            <div className="rounded border-2 border-blue-200 bg-white p-4 shadow-md hover:shadow-lg transition">
              <div className="text-slate-600 text-sm font-medium">Task Completion</div>
              <div className="text-2xl font-bold text-blue-700">{overview?.tasks?.completion_rate ? `${(overview.tasks.completion_rate * 100).toFixed(1)}%` : "-"}</div>
            </div>
          </div>

          <div className="mt-8 rounded border-2 border-blue-200 bg-white p-4 shadow-md">
            <div className="text-slate-900 font-bold mb-2">Leads & Score Trend</div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                  <XAxis dataKey="date" stroke="#64748b" />
                  <YAxis yAxisId="left" stroke="#64748b" />
                  <YAxis yAxisId="right" orientation="right" stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: "#ffffff", border: "2px solid #1e40af", borderRadius: "8px", color: "#0f172a" }} />
                  <Line yAxisId="left" type="monotone" dataKey="leads" stroke="#1e40af" strokeWidth={3} />
                  <Line yAxisId="right" type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="mt-8">
            <ConversionFunnel />
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

