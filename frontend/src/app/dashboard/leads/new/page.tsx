"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function NewLeadsPage() {
  const [showAddLeadModal, setShowAddLeadModal] = useState(false);
  const [leads] = useState([
    { id: "LD-001", name: "Alice Johnson", email: "alice@example.com", phone: "+1 (555) 123-4567", source: "Website", date: "2024-10-22", value: "$5,000" },
    { id: "LD-002", name: "Bob Smith", email: "bob@example.com", phone: "+1 (555) 234-5678", source: "Referral", date: "2024-10-21", value: "$7,500" },
    { id: "LD-003", name: "Carol Davis", email: "carol@example.com", phone: "+1 (555) 345-6789", source: "Ad Campaign", date: "2024-10-20", value: "$6,000" },
    { id: "LD-004", name: "David Wilson", email: "david@example.com", phone: "+1 (555) 456-7890", source: "Website", date: "2024-10-19", value: "$8,500" },
    { id: "LD-005", name: "Emma Brown", email: "emma@example.com", phone: "+1 (555) 567-8901", source: "Phone", date: "2024-10-18", value: "$4,500" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">New Leads</h1>
            <p className="text-slate-600 font-medium mt-1">Recently added leads waiting for follow-up</p>
          </div>
          <button
            onClick={() => setShowAddLeadModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + Add Lead
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">New This Week</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">24</p>
            <p className="text-xs text-slate-600 font-medium mt-2">↑ 12% vs last week</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Lead Value</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">$6,300</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Total: $151.2K</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Top Source</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">Website</p>
            <p className="text-xs text-slate-600 font-medium mt-2">45% of new leads</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Conversion Rate</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">32%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">To qualified</p>
          </div>
        </div>

        {/* New Leads Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent New Leads</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Lead ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Phone</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Source</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Value</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Date Added</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {leads.map((lead) => (
                  <tr key={lead.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{lead.id}</td>
                    <td className="p-4 font-medium text-slate-900">{lead.name}</td>
                    <td className="p-4 text-slate-700">{lead.email}</td>
                    <td className="p-4 text-slate-700">{lead.phone}</td>
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-700">
                        {lead.source}
                      </span>
                    </td>
                    <td className="p-4 font-bold text-slate-900">{lead.value}</td>
                    <td className="p-4 text-slate-700">{lead.date}</td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Contact</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Qualify</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Add Lead Modal */}
      {showAddLeadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowAddLeadModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-2xl font-bold text-slate-900 mb-4">➕ Add New Lead</h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Full Name *</label>
                  <input
                    type="text"
                    placeholder="John Smith"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Email *</label>
                  <input
                    type="email"
                    placeholder="john@example.com"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Phone</label>
                  <input
                    type="tel"
                    placeholder="+1 (555) 123-4567"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Lead Source</label>
                  <select className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600">
                    <option value="">Select source...</option>
                    <option value="website">🌐 Website</option>
                    <option value="referral">👥 Referral</option>
                    <option value="phone">📞 Phone</option>
                    <option value="email">📧 Email</option>
                    <option value="social">📱 Social Media</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Insurance Interest</label>
                <select className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600">
                  <option value="">Select insurance type...</option>
                  <option value="auto">🚗 Auto Insurance</option>
                  <option value="home">🏠 Home Insurance</option>
                  <option value="life">❤️ Life Insurance</option>
                  <option value="health">⚕️ Health Insurance</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Estimated Value</label>
                <input
                  type="text"
                  placeholder="$5,000"
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Notes</label>
                <textarea
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  rows={3}
                  placeholder="Additional information about the lead..."
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAddLeadModal(false)}
                className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowAddLeadModal(false);
                  // In a real app, this would create the lead
                }}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Add Lead
              </button>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}

