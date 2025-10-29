"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function SignedDocumentsPage() {
  const [documents] = useState([
    { id: 1, title: "Home Insurance Policy", customer: "Sarah Johnson", signed: "2024-10-18", signedBy: "Sarah Johnson", status: "Archived" },
    { id: 2, title: "Health Insurance Enrollment", customer: "Emily Davis", signed: "2024-10-16", signedBy: "Emily Davis", status: "Active" },
    { id: 3, title: "Auto Insurance Agreement", customer: "Michael Brown", signed: "2024-10-14", signedBy: "Michael Brown", status: "Active" },
    { id: 4, title: "Life Insurance Rider", customer: "David Wilson", signed: "2024-10-12", signedBy: "David Wilson", status: "Active" },
    { id: 5, title: "Policy Amendment", customer: "Lisa Anderson", signed: "2024-10-10", signedBy: "Lisa Anderson", status: "Archived" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Signed Documents</h1>
            <p className="text-slate-600 font-medium mt-1">Completed and signed documents</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Export Documents
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Signed</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">1,248</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All time</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">This Month</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">156</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Signed documents</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Signature Time</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">2.1 days</p>
            <p className="text-xs text-slate-600 font-medium mt-2">From sent</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Documents</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">892</p>
            <p className="text-xs text-slate-600 font-medium mt-2">In use</p>
          </div>
        </div>

        {/* Signed Documents Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Signed Documents Archive</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Document ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Title</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Signed Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Signed By</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">DOC-{doc.id}</td>
                    <td className="p-4 font-medium text-slate-900">{doc.title}</td>
                    <td className="p-4 text-slate-700">{doc.customer}</td>
                    <td className="p-4 text-slate-700">{doc.signed}</td>
                    <td className="p-4 text-slate-700">{doc.signedBy}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        doc.status === "Active"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-slate-100 text-slate-700"
                      }`}>
                        {doc.status}
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Download</button>
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

