"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function SMSPage() {
  const [messages] = useState<any[]>([]);

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
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No SMS data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Delivery Rate</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No delivery data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Response Rate</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No response data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Response Time</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No response data</p>
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
                {messages.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="p-4 text-center text-slate-600 font-medium">No SMS messages available.</td>
                  </tr>
                ) : (
                  messages.map((msg) => (
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

