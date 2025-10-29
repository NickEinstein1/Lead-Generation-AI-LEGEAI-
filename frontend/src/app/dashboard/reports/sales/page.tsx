"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function SalesReportsPage() {
  const [salesData] = useState([
    { id: 1, agent: "John Smith", policies: 24, revenue: "$156,800", target: "$150,000", achievement: "104.5%" },
    { id: 2, agent: "Sarah Johnson", policies: 18, revenue: "$124,200", target: "$150,000", achievement: "82.8%" },
    { id: 3, agent: "Michael Brown", policies: 31, revenue: "$198,400", target: "$150,000", achievement: "132.3%" },
    { id: 4, agent: "Emily Davis", policies: 15, revenue: "$98,500", target: "$150,000", achievement: "65.7%" },
    { id: 5, agent: "David Wilson", policies: 22, revenue: "$142,300", target: "$150,000", achievement: "94.9%" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Sales Reports</h1>
            <p className="text-slate-600 font-medium mt-1">Track sales performance and revenue metrics</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Export Report
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Revenue</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">$720.2K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">This month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Policies Sold</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">110</p>
            <p className="text-xs text-slate-600 font-medium mt-2">This month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Policy Value</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$6,547</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Per policy</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Target Achievement</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">96.0%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Team average</p>
          </div>
        </div>

        {/* Sales Data Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Agent Sales Performance</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Agent</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Policies Sold</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Revenue</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Target</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Achievement %</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {salesData.map((data) => (
                  <tr key={data.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-medium text-slate-900">{data.agent}</td>
                    <td className="p-4 font-bold text-blue-700">{data.policies}</td>
                    <td className="p-4 font-bold text-slate-900">{data.revenue}</td>
                    <td className="p-4 text-slate-700">{data.target}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        parseFloat(data.achievement) >= 100
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {data.achievement}
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Details</button>
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

