"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function QualifiedLeadsPage() {
  const [leads] = useState([
    { id: "LD-045", name: "Frank Miller", email: "frank@example.com", phone: "+1 (555) 678-9012", score: 92, status: "Hot", value: "$12,000" },
    { id: "LD-046", name: "Grace Lee", email: "grace@example.com", phone: "+1 (555) 789-0123", score: 88, status: "Hot", value: "$10,500" },
    { id: "LD-047", name: "Henry Taylor", email: "henry@example.com", phone: "+1 (555) 890-1234", score: 75, status: "Warm", value: "$8,000" },
    { id: "LD-048", name: "Iris Martinez", email: "iris@example.com", phone: "+1 (555) 901-2345", score: 82, status: "Hot", value: "$11,000" },
    { id: "LD-049", name: "Jack Anderson", email: "jack@example.com", phone: "+1 (555) 012-3456", score: 70, status: "Warm", value: "$7,500" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Qualified Leads</h1>
            <p className="text-slate-600 font-medium mt-1">High-potential leads ready for conversion</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Qualified Lead
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Qualified</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">156</p>
            <p className="text-xs text-slate-600 font-medium mt-2">↑ 8% vs last month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Hot Leads</p>
            <p className="text-3xl font-bold text-red-600 mt-2">48</p>
            <p className="text-xs text-slate-600 font-medium mt-2">30.8% of qualified</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Lead Score</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">81.4</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Out of 100</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Pipeline Value</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$1.2M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $7,692</p>
          </div>
        </div>

        {/* Qualified Leads Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Qualified Leads Pipeline</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Lead ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Phone</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Score</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Value</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {leads.map((lead) => (
                  <tr key={lead.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{lead.id}</td>
                    <td className="p-4 font-medium text-slate-900">{lead.name}</td>
                    <td className="p-4 text-slate-700">{lead.email}</td>
                    <td className="p-4 text-slate-700">{lead.phone}</td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-slate-200 rounded-full h-2">
                          <div className="bg-emerald-600 h-2 rounded-full" style={{ width: `${lead.score}%` }}></div>
                        </div>
                        <span className="font-bold text-slate-900">{lead.score}</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        lead.status === "Hot"
                          ? "bg-red-100 text-red-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {lead.status}
                      </span>
                    </td>
                    <td className="p-4 font-bold text-slate-900">{lead.value}</td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Convert</button>
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

