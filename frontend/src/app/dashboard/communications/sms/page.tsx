"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function SMSPage() {
  const [messages] = useState([
    { id: 1, to: "+1 (555) 123-4567", message: "Your policy renewal is due on 2024-11-15", sent: "2024-10-22", status: "Delivered", response: "Yes" },
    { id: 2, to: "+1 (555) 234-5678", message: "Claim approved! Check your email for details.", sent: "2024-10-21", status: "Delivered", response: "No" },
    { id: 3, to: "+1 (555) 345-6789", message: "Welcome to our insurance platform!", sent: "2024-10-20", status: "Delivered", response: "Yes" },
    { id: 4, to: "+1 (555) 456-7890", message: "Special offer: 15% discount on home insurance", sent: "2024-10-19", status: "Delivered", response: "No" },
    { id: 5, to: "+1 (555) 567-8901", message: "Your document is ready for signature", sent: "2024-10-18", status: "Delivered", response: "Yes" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">SMS Communications</h1>
            <p className="text-slate-600 font-medium mt-1">Track and manage SMS messages</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Send SMS
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total SMS Sent</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">1,234</p>
            <p className="text-xs text-slate-600 font-medium mt-2">This month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Delivery Rate</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">99.2%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Successfully delivered</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Response Rate</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">67.8%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Average</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Response Time</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">4.2 min</p>
            <p className="text-xs text-slate-600 font-medium mt-2">From delivery</p>
          </div>
        </div>

        {/* SMS Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent SMS Messages</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Message ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">To</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Message</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Sent Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Response</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {messages.map((msg) => (
                  <tr key={msg.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">SMS-{msg.id}</td>
                    <td className="p-4 text-slate-700">{msg.to}</td>
                    <td className="p-4 font-medium text-slate-900 truncate">{msg.message}</td>
                    <td className="p-4 text-slate-700">{msg.sent}</td>
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                        {msg.status}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        msg.response === "Yes"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-slate-100 text-slate-700"
                      }`}>
                        {msg.response}
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Reply</button>
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

