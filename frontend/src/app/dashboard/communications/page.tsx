"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function CommunicationsPage() {
  const [communications] = useState([
    { id: 1, type: "email", customer: "John Smith", subject: "Policy Renewal Reminder", status: "sent", date: "2024-10-22", channel: "Email" },
    { id: 2, type: "sms", customer: "Sarah Johnson", subject: "Claim Status Update", status: "delivered", date: "2024-10-21", channel: "SMS" },
    { id: 3, type: "call", customer: "Michael Brown", subject: "Follow-up Call", status: "completed", date: "2024-10-20", channel: "Phone" },
    { id: 4, type: "email", customer: "Emily Davis", subject: "New Product Offer", status: "sent", date: "2024-10-19", channel: "Email" },
    { id: 5, type: "sms", customer: "David Wilson", subject: "Payment Reminder", status: "delivered", date: "2024-10-18", channel: "SMS" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Communications</h1>
            <p className="text-slate-600 font-medium mt-1">Manage customer communications and campaigns</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Communication
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Sent</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">3,456</p>
            <p className="text-xs text-slate-600 font-medium mt-2">This month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Delivery Rate</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">98.5%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Successfully delivered</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Open Rate</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">42.3%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Email opens</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Click Rate</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">18.7%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Link clicks</p>
          </div>
        </div>

        {/* Communication Channels */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">📧 Emails</p>
            <p className="text-2xl font-bold text-blue-700 mt-2">1,234</p>
            <p className="text-xs text-slate-600 font-medium mt-2">35.7% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">💬 SMS</p>
            <p className="text-2xl font-bold text-emerald-700 mt-2">1,567</p>
            <p className="text-xs text-slate-600 font-medium mt-2">45.3% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">☎️ Calls</p>
            <p className="text-2xl font-bold text-red-700 mt-2">456</p>
            <p className="text-xs text-slate-600 font-medium mt-2">13.2% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">📢 Campaigns</p>
            <p className="text-2xl font-bold text-purple-700 mt-2">45</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Active campaigns</p>
          </div>
        </div>

        {/* Communications Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Communications</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Subject</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Channel</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {communications.map((comm) => (
                  <tr key={comm.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4">
                      <span className="text-lg">
                        {comm.type === "email" ? "📧" : comm.type === "sms" ? "💬" : "☎️"}
                      </span>
                    </td>
                    <td className="p-4 font-medium text-slate-900">{comm.customer}</td>
                    <td className="p-4 text-slate-700">{comm.subject}</td>
                    <td className="p-4 text-slate-700">{comm.channel}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        comm.status === "sent" || comm.status === "delivered" || comm.status === "completed"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {comm.status === "sent" ? "✓ Sent" : comm.status === "delivered" ? "✓ Delivered" : "✓ Completed"}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{comm.date}</td>
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

