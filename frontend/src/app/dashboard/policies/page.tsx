"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function PoliciesPage() {
  const [policies] = useState([
    { id: "POL-001", customer: "John Smith", type: "Auto", status: "active", premium: "$1,200/yr", startDate: "2024-01-15", endDate: "2025-01-15" },
    { id: "POL-002", customer: "Sarah Johnson", type: "Home", status: "active", premium: "$1,500/yr", startDate: "2024-02-01", endDate: "2025-02-01" },
    { id: "POL-003", customer: "Michael Brown", type: "Life", status: "expired", premium: "$500/yr", startDate: "2023-06-01", endDate: "2024-06-01" },
    { id: "POL-004", customer: "Emily Davis", type: "Health", status: "active", premium: "$2,000/yr", startDate: "2024-03-10", endDate: "2025-03-10" },
    { id: "POL-005", customer: "David Wilson", type: "Auto", status: "active", premium: "$1,100/yr", startDate: "2024-04-20", endDate: "2025-04-20" },
  ]);

  const typeColors: { [key: string]: string } = {
    "Auto": "bg-blue-100 text-blue-700",
    "Home": "bg-amber-100 text-amber-700",
    "Life": "bg-red-100 text-red-700",
    "Health": "bg-emerald-100 text-emerald-700",
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Policies</h1>
            <p className="text-slate-600 font-medium mt-1">Manage all insurance policies</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Policy
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Policies</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">156</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All types</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">148</p>
            <p className="text-xs text-slate-600 font-medium mt-2">94.9% active</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Expiring Soon</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">12</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Next 30 days</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Annual Revenue</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$187K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $1,200</p>
          </div>
        </div>

        {/* Policy Types */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">🚗 Auto Insurance</p>
            <p className="text-2xl font-bold text-blue-700 mt-2">42</p>
            <p className="text-xs text-slate-600 font-medium mt-2">26.9% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">🏠 Home Insurance</p>
            <p className="text-2xl font-bold text-amber-700 mt-2">38</p>
            <p className="text-xs text-slate-600 font-medium mt-2">24.4% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">❤️ Life Insurance</p>
            <p className="text-2xl font-bold text-red-700 mt-2">35</p>
            <p className="text-xs text-slate-600 font-medium mt-2">22.4% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">⚕️ Health Insurance</p>
            <p className="text-2xl font-bold text-emerald-700 mt-2">41</p>
            <p className="text-xs text-slate-600 font-medium mt-2">26.3% of total</p>
          </div>
        </div>

        {/* Policies Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">All Policies</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Policy ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Premium</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Start Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">End Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((policy) => (
                  <tr key={policy.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{policy.id}</td>
                    <td className="p-4 font-medium text-slate-900">{policy.customer}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${typeColors[policy.type]}`}>
                        {policy.type}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        policy.status === "active"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-red-100 text-red-700"
                      }`}>
                        {policy.status === "active" ? "✓ Active" : "✗ Expired"}
                      </span>
                    </td>
                    <td className="p-4 font-bold text-slate-900">{policy.premium}</td>
                    <td className="p-4 text-slate-700">{policy.startDate}</td>
                    <td className="p-4 text-slate-700">{policy.endDate}</td>
                    <td className="p-4">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
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

