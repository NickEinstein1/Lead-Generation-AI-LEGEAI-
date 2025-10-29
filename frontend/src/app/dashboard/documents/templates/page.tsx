"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function DocumentTemplatesPage() {
  const [templates] = useState([
    { id: 1, name: "Auto Insurance Policy", category: "Auto", created: "2024-01-15", lastUsed: "2024-10-20", usage: 245 },
    { id: 2, name: "Home Insurance Agreement", category: "Home", created: "2024-01-10", lastUsed: "2024-10-19", usage: 189 },
    { id: 3, name: "Health Insurance Enrollment", category: "Health", created: "2024-02-01", lastUsed: "2024-10-18", usage: 156 },
    { id: 4, name: "Life Insurance Application", category: "Life", created: "2024-01-20", lastUsed: "2024-10-17", usage: 98 },
    { id: 5, name: "Policy Amendment Form", category: "General", created: "2024-03-05", lastUsed: "2024-10-16", usage: 67 },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Document Templates</h1>
            <p className="text-slate-600 font-medium mt-1">Manage and organize document templates</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Create Template
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Templates</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">28</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Active templates</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Usage</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">3,245</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All time</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Most Used</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">245</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Auto Insurance</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Last Updated</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">2 days</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Ago</p>
          </div>
        </div>

        {/* Templates Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Available Templates</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Template ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Category</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Created</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Last Used</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Usage Count</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {templates.map((template) => (
                  <tr key={template.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">TMPL-{template.id}</td>
                    <td className="p-4 font-medium text-slate-900">{template.name}</td>
                    <td className="p-4 text-slate-700">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-700">
                        {template.category}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{template.created}</td>
                    <td className="p-4 text-slate-700">{template.lastUsed}</td>
                    <td className="p-4 font-bold text-slate-900">{template.usage}</td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Use</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Edit</button>
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

