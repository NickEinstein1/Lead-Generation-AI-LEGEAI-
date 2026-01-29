"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function SalesReportsPage() {
  const [salesData] = useState<any[]>([]);

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
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No revenue data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Policies Sold</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No sales data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Policy Value</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No value data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Target Achievement</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No target data</p>
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
                {salesData.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="p-4 text-center text-slate-600 font-medium">No sales data available.</td>
                  </tr>
                ) : (
                  salesData.map((data) => (
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
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

