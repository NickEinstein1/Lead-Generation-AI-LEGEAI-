"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function ReportsPage() {
  const [selectedReport, setSelectedReport] = useState<any>(null);
  const [reports] = useState([
    { id: 1, name: "Monthly Sales Report", type: "Sales", created: "2024-10-20", period: "October 2024", status: "ready" },
    { id: 2, name: "Pipeline Analysis", type: "Pipeline", created: "2024-10-19", period: "Q4 2024", status: "ready" },
    { id: 3, name: "Agent Performance", type: "Performance", created: "2024-10-18", period: "October 2024", status: "ready" },
    { id: 4, name: "Customer Retention", type: "Analytics", created: "2024-10-17", period: "YTD 2024", status: "ready" },
    { id: 5, name: "Claims Analysis", type: "Claims", created: "2024-10-16", period: "October 2024", status: "ready" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Reports</h1>
            <p className="text-slate-600 font-medium mt-1">Generate and view business reports</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Generate Report
          </button>
        </div>

        {/* Quick Report Types */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <button className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg hover:border-blue-400 transition text-left">
            <p className="text-2xl mb-2">💰</p>
            <p className="font-bold text-slate-900">Sales Report</p>
            <p className="text-xs text-slate-600 mt-2">Revenue and deals</p>
          </button>
          <button className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg hover:border-blue-400 transition text-left">
            <p className="text-2xl mb-2">📊</p>
            <p className="font-bold text-slate-900">Pipeline Report</p>
            <p className="text-xs text-slate-600 mt-2">Lead progression</p>
          </button>
          <button className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg hover:border-blue-400 transition text-left">
            <p className="text-2xl mb-2">⚡</p>
            <p className="font-bold text-slate-900">Performance Report</p>
            <p className="text-xs text-slate-600 mt-2">Agent metrics</p>
          </button>
          <button className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg hover:border-blue-400 transition text-left">
            <p className="text-2xl mb-2">📈</p>
            <p className="font-bold text-slate-900">Analytics Report</p>
            <p className="text-xs text-slate-600 mt-2">Trends & insights</p>
          </button>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Revenue</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">$2.4M</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 12% vs last month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Deals Closed</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">156</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 8% vs last month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Deal Size</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$15.4K</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 3% vs last month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Win Rate</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">42.5%</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 2.1% vs last month</p>
          </div>
        </div>

        {/* Reports Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Reports</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Report Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Period</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Created</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((report) => (
                  <tr key={report.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-medium text-slate-900">{report.name}</td>
                    <td className="p-4 text-slate-700">{report.type}</td>
                    <td className="p-4 text-slate-700">{report.period}</td>
                    <td className="p-4 text-slate-700">{report.created}</td>
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                        ✓ Ready
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button
                        onClick={() => setSelectedReport(report)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm hover:underline">Download</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* View Report Details Modal */}
        {selectedReport && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedReport(null)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">📊 Report Details</h2>
                <button onClick={() => setSelectedReport(null)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Report ID</label>
                    <p className="text-lg font-bold text-blue-700">{selectedReport.id}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Status</label>
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                      ✓ Ready
                    </span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Report Name</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedReport.name}</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Report Type</label>
                    <p className="text-lg text-slate-700">{selectedReport.type}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Period</label>
                    <p className="text-lg text-slate-700">{selectedReport.period}</p>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Generated Date</label>
                  <p className="text-lg text-slate-700">{selectedReport.created}</p>
                </div>

                <div className="pt-4 border-t border-slate-200">
                  <div className="flex gap-3">
                    <button
                      onClick={() => setSelectedReport(null)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => alert('Download report functionality coming soon!')}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Download Report
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

