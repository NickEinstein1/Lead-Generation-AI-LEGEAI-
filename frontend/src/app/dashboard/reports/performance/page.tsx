"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function PerformanceReportsPage() {
  const [performanceData] = useState([
    { id: 1, metric: "Lead Response Time", value: "2.3 hrs", target: "4 hrs", status: "Exceeding" },
    { id: 2, metric: "Customer Satisfaction", value: "4.6/5", target: "4.0/5", status: "Exceeding" },
    { id: 3, metric: "Policy Renewal Rate", value: "94%", target: "90%", status: "Exceeding" },
    { id: 4, metric: "Claims Processing Time", value: "3.2 days", target: "5 days", status: "Exceeding" },
    { id: 5, metric: "Agent Productivity", value: "24 policies/mo", target: "20 policies/mo", status: "Exceeding" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Performance Reports</h1>
            <p className="text-slate-600 font-medium mt-1">Monitor key performance indicators and metrics</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Export Report
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Overall Performance</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">98.5%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Target achievement</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">KPIs Exceeding Target</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">5/5</p>
            <p className="text-xs text-slate-600 font-medium mt-2">100% on track</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Team Efficiency</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">92.3%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Utilization rate</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Quality Score</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">9.2/10</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Average</p>
          </div>
        </div>

        {/* Performance Data Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Key Performance Indicators</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Metric</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Current Value</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Target</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {performanceData.map((data) => (
                  <tr key={data.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-medium text-slate-900">{data.metric}</td>
                    <td className="p-4 font-bold text-blue-700">{data.value}</td>
                    <td className="p-4 text-slate-700">{data.target}</td>
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                        {data.status}
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Trend</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

