"use client";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { getDashboardOverview, getLeadTimeseries, getScoreTimeseries } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

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
    <div className="min-h-screen bg-gradient-to-b from-[#f8fafc] to-[#eef2f7]">
      <div className="relative overflow-hidden">
        <div className="absolute right-0 top-0 h-40 w-40 opacity-30 blur-2xl rounded-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-300/30 via-blue-400/10 to-transparent" />
        <div className="mx-auto max-w-6xl px-6 py-8">
          <div className="flex items-center justify-between gap-6">
            <div>
              <h1 className="text-2xl font-semibold text-neutral-900">Analytics</h1>
              <p className="text-neutral-600">Funnel, sources, and model performance at a glance.</p>
            </div>
            <div className="h-16 w-32 opacity-70">
              <MiniAccent3D />
            </div>
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded border bg-white p-4 shadow-sm">
              <div className="text-neutral-500 text-sm">Assignments (total)</div>
              <div className="text-2xl font-semibold">{overview?.routing?.total_assignments ?? 0}</div>
            </div>
            <div className="rounded border bg-white p-4 shadow-sm">
              <div className="text-neutral-500 text-sm">Avg Assignments/Rep</div>
              <div className="text-2xl font-semibold">{overview?.routing?.average_assignments ?? 0}</div>
            </div>
            <div className="rounded border bg-white p-4 shadow-sm">
              <div className="text-neutral-500 text-sm">Task Completion</div>
              <div className="text-2xl font-semibold">{overview?.tasks?.completion_rate ? `${(overview.tasks.completion_rate * 100).toFixed(1)}%` : "-"}</div>
            </div>
          </div>

          <div className="mt-8 rounded border bg-white p-4 shadow-sm">
            <div className="text-neutral-900 font-medium mb-2">Leads & Score Trend</div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Line yAxisId="left" type="monotone" dataKey="leads" stroke="#2563eb" strokeWidth={2} />
                  <Line yAxisId="right" type="monotone" dataKey="score" stroke="#60a5fa" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

