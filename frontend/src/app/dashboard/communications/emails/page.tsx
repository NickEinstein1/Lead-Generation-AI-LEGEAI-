"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function EmailsPage() {
  const [emails] = useState([
    { id: 1, to: "john@example.com", subject: "Your Auto Insurance Policy Renewal", sent: "2024-10-22", status: "Delivered", opens: 1 },
    { id: 2, to: "sarah@example.com", subject: "Welcome to Our Insurance Platform", sent: "2024-10-21", status: "Delivered", opens: 2 },
    { id: 3, to: "michael@example.com", subject: "Claim Status Update", sent: "2024-10-20", status: "Delivered", opens: 1 },
    { id: 4, to: "emily@example.com", subject: "Special Offer: Home Insurance Discount", sent: "2024-10-19", status: "Delivered", opens: 0 },
    { id: 5, to: "david@example.com", subject: "Policy Amendment Confirmation", sent: "2024-10-18", status: "Delivered", opens: 1 },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Email Communications</h1>
            <p className="text-slate-600 font-medium mt-1">Track and manage email communications</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Send Email
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Emails Sent</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">2,456</p>
            <p className="text-xs text-slate-600 font-medium mt-2">This month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Delivery Rate</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">98.7%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Successfully delivered</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Open Rate</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">42.3%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Average</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Click Rate</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">18.5%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Average</p>
          </div>
        </div>

        {/* Emails Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Emails</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Email ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">To</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Subject</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Sent Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Opens</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {emails.map((email) => (
                  <tr key={email.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">EMAIL-{email.id}</td>
                    <td className="p-4 text-slate-700">{email.to}</td>
                    <td className="p-4 font-medium text-slate-900">{email.subject}</td>
                    <td className="p-4 text-slate-700">{email.sent}</td>
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                        {email.status}
                      </span>
                    </td>
                    <td className="p-4 font-bold text-slate-900">{email.opens}</td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Resend</button>
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

