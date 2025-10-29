"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function ClaimsPage() {
  const [claims] = useState([
    { id: "CLM-001", policy: "POL-001", customer: "John Smith", type: "Auto", amount: "$5,000", status: "approved", date: "2024-10-15", dueDate: "2024-11-15" },
    { id: "CLM-002", policy: "POL-002", customer: "Sarah Johnson", type: "Home", amount: "$12,500", status: "pending", date: "2024-10-20", dueDate: "2024-11-20" },
    { id: "CLM-003", policy: "POL-004", customer: "Emily Davis", type: "Health", amount: "$2,300", status: "approved", date: "2024-10-10", dueDate: "2024-11-10" },
    { id: "CLM-004", policy: "POL-005", customer: "David Wilson", type: "Auto", amount: "$8,750", status: "rejected", date: "2024-10-05", dueDate: "2024-11-05" },
    { id: "CLM-005", policy: "POL-001", customer: "John Smith", type: "Auto", amount: "$3,200", status: "pending", date: "2024-10-22", dueDate: "2024-11-22" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Claims</h1>
            <p className="text-slate-600 font-medium mt-1">Manage insurance claims and payouts</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Claim
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Claims</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">248</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All time</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Pending</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">8</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Awaiting review</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Approved</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">215</p>
            <p className="text-xs text-slate-600 font-medium mt-2">86.7% approval rate</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Payout</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$1.2M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $4,839</p>
          </div>
        </div>

        {/* Claims by Status */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">✓ Approved</p>
            <p className="text-2xl font-bold text-emerald-600 mt-2">215</p>
            <p className="text-xs text-slate-600 font-medium mt-2">$1.04M total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">⏳ Pending</p>
            <p className="text-2xl font-bold text-amber-600 mt-2">8</p>
            <p className="text-xs text-slate-600 font-medium mt-2">$31.75K total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">✗ Rejected</p>
            <p className="text-2xl font-bold text-red-600 mt-2">25</p>
            <p className="text-xs text-slate-600 font-medium mt-2">$156.25K total</p>
          </div>
        </div>

        {/* Claims Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Claims</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Claim ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Policy</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Amount</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {claims.map((claim) => (
                  <tr key={claim.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{claim.id}</td>
                    <td className="p-4 font-medium text-slate-900">{claim.policy}</td>
                    <td className="p-4 text-slate-700">{claim.customer}</td>
                    <td className="p-4 text-slate-700">{claim.type}</td>
                    <td className="p-4 font-bold text-slate-900">{claim.amount}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        claim.status === "approved"
                          ? "bg-emerald-100 text-emerald-700"
                          : claim.status === "pending"
                          ? "bg-amber-100 text-amber-700"
                          : "bg-red-100 text-red-700"
                      }`}>
                        {claim.status === "approved" ? "✓ Approved" : claim.status === "pending" ? "⏳ Pending" : "✗ Rejected"}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{claim.date}</td>
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

