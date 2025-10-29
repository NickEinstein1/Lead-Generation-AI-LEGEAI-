"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function PendingClaimsPage() {
  const [claims] = useState([
    { id: "CLM-PEND-001", policy: "POL-001", customer: "John Smith", type: "Auto", amount: "$5,000", submitted: "2024-10-20", daysWaiting: 2 },
    { id: "CLM-PEND-002", policy: "POL-002", customer: "Sarah Johnson", type: "Home", amount: "$12,500", submitted: "2024-10-18", daysWaiting: 4 },
    { id: "CLM-PEND-003", policy: "POL-004", customer: "Emily Davis", type: "Health", amount: "$2,300", submitted: "2024-10-15", daysWaiting: 7 },
    { id: "CLM-PEND-004", policy: "POL-005", customer: "David Wilson", type: "Auto", amount: "$8,750", submitted: "2024-10-12", daysWaiting: 10 },
    { id: "CLM-PEND-005", policy: "POL-001", customer: "John Smith", amount: "$3,200", submitted: "2024-10-22", daysWaiting: 0 },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Pending Claims</h1>
            <p className="text-slate-600 font-medium mt-1">Claims awaiting review and approval</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Review Claims
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Pending Claims</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">8</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Awaiting review</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Amount</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$31.75K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $3,969</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Wait Time</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">5.2 days</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Since submission</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Urgent (7+ days)</p>
            <p className="text-3xl font-bold text-red-600 mt-2">2</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Needs attention</p>
          </div>
        </div>

        {/* Pending Claims Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Pending Claims Review</h2>
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
                  <th className="text-left p-4 text-slate-900 font-bold">Days Waiting</th>
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
                        claim.daysWaiting >= 7
                          ? "bg-red-100 text-red-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {claim.daysWaiting} days
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Review</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Approve</button>
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

