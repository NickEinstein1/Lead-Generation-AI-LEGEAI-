"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function RejectedClaimsPage() {
  const [claims] = useState([
    { id: "CLM-REJ-001", policy: "POL-006", customer: "Lisa Anderson", type: "Auto", amount: "$4,500", rejected: "2024-10-10", reason: "Policy Lapsed" },
    { id: "CLM-REJ-002", policy: "POL-007", customer: "James Wilson", type: "Home", amount: "$8,000", rejected: "2024-10-08", reason: "Exclusion Applied" },
    { id: "CLM-REJ-003", policy: "POL-008", customer: "Patricia Moore", type: "Health", amount: "$1,200", rejected: "2024-10-05", reason: "Pre-existing Condition" },
    { id: "CLM-REJ-004", policy: "POL-009", customer: "Robert Taylor", type: "Auto", amount: "$6,250", rejected: "2024-10-02", reason: "Fraud Suspected" },
    { id: "CLM-REJ-005", policy: "POL-010", customer: "Jennifer White", type: "Life", amount: "$15,000", rejected: "2024-09-28", reason: "Insufficient Documentation" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Rejected Claims</h1>
            <p className="text-slate-600 font-medium mt-1">Claims that have been denied or rejected</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Appeal Claim
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Rejected Claims</p>
            <p className="text-3xl font-bold text-red-600 mt-2">25</p>
            <p className="text-xs text-slate-600 font-medium mt-2">10.1% rejection rate</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Amount</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$156.25K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $6,250</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Appeals Filed</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">3</p>
            <p className="text-xs text-slate-600 font-medium mt-2">12% appeal rate</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Top Reason</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">Policy Lapsed</p>
            <p className="text-xs text-slate-600 font-medium mt-2">40% of rejections</p>
          </div>
        </div>

        {/* Rejected Claims Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Rejected Claims Details</h2>
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
                  <th className="text-left p-4 text-slate-900 font-bold">Rejection Reason</th>
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
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-red-100 text-red-700">
                        {claim.reason}
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Appeal</button>
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

