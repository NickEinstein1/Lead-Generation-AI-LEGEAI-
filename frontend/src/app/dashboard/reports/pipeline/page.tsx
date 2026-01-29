"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function PipelineReportsPage() {
  const [pipelineData] = useState<any[]>([]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Pipeline Reports</h1>
            <p className="text-slate-600 font-medium mt-1">Analyze sales pipeline and conversion metrics</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Export Report
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Pipeline Value</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No pipeline data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Opportunities</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No opportunity data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Deal Size</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No deal size data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Overall Conversion</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No conversion data</p>
          </div>
        </div>

        {/* Pipeline Data Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Pipeline Stage Analysis</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Stage</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Opportunities</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Total Value</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Avg Deal Size</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Conversion Rate</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {pipelineData.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="p-4 text-center text-slate-600 font-medium">No pipeline data available.</td>
                  </tr>
                ) : (
                  pipelineData.map((data) => (
                    <tr key={data.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                      <td className="p-4 font-medium text-slate-900">{data.stage}</td>
                      <td className="p-4 font-bold text-blue-700">{data.count}</td>
                      <td className="p-4 font-bold text-slate-900">{data.value}</td>
                      <td className="p-4 text-slate-700">{data.avgValue}</td>
                      <td className="p-4">
                        <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                          {data.conversionRate}
                        </span>
                      </td>
                      <td className="p-4 space-x-2">
                        <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
                        <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Analyze</button>
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

