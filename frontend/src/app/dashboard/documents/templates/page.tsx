"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function DocumentTemplatesPage() {
  const [templates] = useState<any[]>([]);

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
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No template data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Usage</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No usage data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Most Used</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No usage data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Last Updated</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No update data</p>
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
                {templates.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="p-4 text-center text-slate-600 font-medium">No templates available.</td>
                  </tr>
                ) : (
                  templates.map((template) => (
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

