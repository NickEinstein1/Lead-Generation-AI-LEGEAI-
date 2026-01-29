"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function IntegrationsPage() {
  const [integrations] = useState<any[]>([]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Integrations</h1>
            <p className="text-slate-600 font-medium mt-1">Manage third-party integrations and connections</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Add Integration
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Integrations</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No integration data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Connected</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No status data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Disconnected</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No status data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Last Sync</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No sync data</p>
          </div>
        </div>

        {/* Integrations Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Connected Services</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Service</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Last Sync</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {integrations.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="p-4 text-center text-slate-600 font-medium">No integrations available.</td>
                  </tr>
                ) : (
                  integrations.map((integration) => (
                    <tr key={integration.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                      <td className="p-4 font-medium text-slate-900">
                        <span className="mr-2">{integration.icon}</span>
                        {integration.name}
                      </td>
                      <td className="p-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                          integration.status === "Connected"
                            ? "bg-emerald-100 text-emerald-700"
                            : "bg-slate-100 text-slate-700"
                        }`}>
                          {integration.status}
                        </span>
                      </td>
                      <td className="p-4 text-slate-700">{integration.lastSync}</td>
                      <td className="p-4 space-x-2">
                        {integration.status === "Connected" ? (
                          <>
                            <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Settings</button>
                            <button className="text-red-600 hover:text-red-800 font-medium text-sm">Disconnect</button>
                          </>
                        ) : (
                          <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Connect</button>
                        )}
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

