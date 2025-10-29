"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function CampaignsPage() {
  const [campaigns] = useState([
    { id: 1, name: "Fall Insurance Sale", type: "Email", status: "Active", recipients: 2500, sent: 2450, opens: 1038, clicks: 192 },
    { id: 2, name: "New Customer Welcome", type: "Multi-channel", status: "Active", recipients: 1200, sent: 1200, opens: 624, clicks: 156 },
    { id: 3, name: "Policy Renewal Reminder", type: "SMS", status: "Completed", recipients: 3000, sent: 2980, opens: 2383, clicks: 447 },
    { id: 4, name: "Health Insurance Promotion", type: "Email", status: "Scheduled", recipients: 1800, sent: 0, opens: 0, clicks: 0 },
    { id: 5, name: "Referral Program Launch", type: "Multi-channel", status: "Active", recipients: 2200, sent: 2150, opens: 1075, clicks: 258 },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Marketing Campaigns</h1>
            <p className="text-slate-600 font-medium mt-1">Create and manage marketing campaigns</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Campaign
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Campaigns</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">3</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Running now</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Recipients</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">5,900</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Across campaigns</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Open Rate</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">42.1%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All campaigns</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Click Rate</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">8.9%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All campaigns</p>
          </div>
        </div>

        {/* Campaigns Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Campaign Performance</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Campaign ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Recipients</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Sent</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Opens</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Clicks</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {campaigns.map((campaign) => (
                  <tr key={campaign.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">CAMP-{campaign.id}</td>
                    <td className="p-4 font-medium text-slate-900">{campaign.name}</td>
                    <td className="p-4 text-slate-700">{campaign.type}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        campaign.status === "Active"
                          ? "bg-emerald-100 text-emerald-700"
                          : campaign.status === "Completed"
                          ? "bg-slate-100 text-slate-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {campaign.status}
                      </span>
                    </td>
                    <td className="p-4 font-bold text-slate-900">{campaign.recipients}</td>
                    <td className="p-4 font-bold text-slate-900">{campaign.sent}</td>
                    <td className="p-4 font-bold text-slate-900">{campaign.opens}</td>
                    <td className="p-4 font-bold text-slate-900">{campaign.clicks}</td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
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

