"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function ContactedLeadsPage() {
  const [leads] = useState([
    { id: "LD-078", name: "Kevin White", email: "kevin@example.com", phone: "+1 (555) 123-4567", lastContact: "2024-10-22", nextFollowUp: "2024-10-25", status: "Interested" },
    { id: "LD-079", name: "Laura Green", email: "laura@example.com", phone: "+1 (555) 234-5678", lastContact: "2024-10-21", nextFollowUp: "2024-10-24", status: "Considering" },
    { id: "LD-080", name: "Michael Black", email: "michael@example.com", phone: "+1 (555) 345-6789", lastContact: "2024-10-20", nextFollowUp: "2024-10-23", status: "Interested" },
    { id: "LD-081", name: "Nancy Red", email: "nancy@example.com", phone: "+1 (555) 456-7890", lastContact: "2024-10-19", nextFollowUp: "2024-10-22", status: "Not Interested" },
    { id: "LD-082", name: "Oscar Blue", email: "oscar@example.com", phone: "+1 (555) 567-8901", lastContact: "2024-10-18", nextFollowUp: "2024-10-21", status: "Considering" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Contacted Leads</h1>
            <p className="text-slate-600 font-medium mt-1">Leads that have been contacted and are in follow-up</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Log Contact
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Contacted</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">234</p>
            <p className="text-xs text-slate-600 font-medium mt-2">↑ 15% vs last month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Interested</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">89</p>
            <p className="text-xs text-slate-600 font-medium mt-2">38% conversion rate</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Response Time</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">2.3 hrs</p>
            <p className="text-xs text-slate-600 font-medium mt-2">From contact</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Follow-ups Due</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">12</p>
            <p className="text-xs text-slate-600 font-medium mt-2">This week</p>
          </div>
        </div>

        {/* Contacted Leads Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Contact History & Follow-ups</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Lead ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Last Contact</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Next Follow-up</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {leads.map((lead) => (
                  <tr key={lead.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{lead.id}</td>
                    <td className="p-4 font-medium text-slate-900">{lead.name}</td>
                    <td className="p-4 text-slate-700">{lead.email}</td>
                    <td className="p-4 text-slate-700">{lead.lastContact}</td>
                    <td className="p-4 text-slate-700">{lead.nextFollowUp}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        lead.status === "Interested"
                          ? "bg-emerald-100 text-emerald-700"
                          : lead.status === "Considering"
                          ? "bg-amber-100 text-amber-700"
                          : "bg-red-100 text-red-700"
                      }`}>
                        {lead.status}
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Follow-up</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">History</button>
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

