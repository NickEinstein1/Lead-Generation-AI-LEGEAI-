"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function CallsPage() {
  const [calls] = useState<any[]>([]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Call Logs</h1>
            <p className="text-slate-600 font-medium mt-1">Track and manage customer calls</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Make Call
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Calls</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No call data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Call Duration</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No duration data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Completion Rate</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No completion data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Inbound vs Outbound</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No distribution data</p>
          </div>
        </div>

        {/* Calls Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Calls</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Call ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Phone</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Duration</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Outcome</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {calls.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="p-4 text-center text-slate-600 font-medium">No call records available.</td>
                  </tr>
                ) : (
                  calls.map((call) => (
                    <tr key={call.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                      <td className="p-4 font-bold text-blue-700">CALL-{call.id}</td>
                      <td className="p-4 font-medium text-slate-900">{call.to}</td>
                      <td className="p-4 text-slate-700">{call.phone}</td>
                      <td className="p-4 text-slate-700">{call.date}</td>
                      <td className="p-4 font-bold text-slate-900">{call.duration}</td>
                      <td className="p-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                          call.type === "Outbound"
                            ? "bg-blue-100 text-blue-700"
                            : "bg-emerald-100 text-emerald-700"
                        }`}>
                          {call.type}
                        </span>
                      </td>
                      <td className="p-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                          call.outcome === "Completed"
                            ? "bg-emerald-100 text-emerald-700"
                            : "bg-amber-100 text-amber-700"
                        }`}>
                          {call.outcome}
                        </span>
                      </td>
                      <td className="p-4 space-x-2">
                        <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Listen</button>
                        <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Notes</button>
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

