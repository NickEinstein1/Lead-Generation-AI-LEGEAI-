"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function PendingSignaturePage() {
  const [documents] = useState([
    { id: 1, title: "Home Insurance Policy", customer: "Sarah Johnson", created: "2024-10-20", daysWaiting: 2, priority: "High" },
    { id: 2, title: "Health Insurance Enrollment", customer: "Emily Davis", created: "2024-10-18", daysWaiting: 4, priority: "Medium" },
    { id: 3, title: "Auto Insurance Agreement", customer: "Michael Brown", created: "2024-10-15", daysWaiting: 7, priority: "High" },
    { id: 4, title: "Life Insurance Rider", customer: "David Wilson", created: "2024-10-12", daysWaiting: 10, priority: "Medium" },
    { id: 5, title: "Policy Amendment", customer: "Lisa Anderson", created: "2024-10-10", daysWaiting: 12, priority: "High" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Pending Signatures</h1>
            <p className="text-slate-600 font-medium mt-1">Documents awaiting customer signature</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Send for Signature
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Pending Signatures</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">42</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Awaiting action</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Wait Time</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">4.8 days</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Since sent</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Overdue (7+ days)</p>
            <p className="text-3xl font-bold text-red-600 mt-2">8</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Needs follow-up</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Signature Rate</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">94.8%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Completion rate</p>
          </div>
        </div>

        {/* Pending Signatures Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Documents Awaiting Signature</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Document ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Title</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Sent Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Days Waiting</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Priority</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">DOC-{doc.id}</td>
                    <td className="p-4 font-medium text-slate-900">{doc.title}</td>
                    <td className="p-4 text-slate-700">{doc.customer}</td>
                    <td className="p-4 text-slate-700">{doc.created}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        doc.daysWaiting >= 7
                          ? "bg-red-100 text-red-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {doc.daysWaiting} days
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        doc.priority === "High"
                          ? "bg-red-100 text-red-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {doc.priority}
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Remind</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">View</button>
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

