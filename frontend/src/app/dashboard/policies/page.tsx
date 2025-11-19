"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function PoliciesPage() {
  const [showNewPolicyModal, setShowNewPolicyModal] = useState(false);
  const [selectedPolicy, setSelectedPolicy] = useState<any>(null);
  const [policies] = useState([
    { id: "POL-001", customer: "John Smith", type: "Auto", status: "active", premium: "$1,200/yr", startDate: "2024-01-15", endDate: "2025-01-15" },
    { id: "POL-002", customer: "Sarah Johnson", type: "Home", status: "active", premium: "$1,500/yr", startDate: "2024-02-01", endDate: "2025-02-01" },
    { id: "POL-003", customer: "Michael Brown", type: "Life", status: "expired", premium: "$500/yr", startDate: "2023-06-01", endDate: "2024-06-01" },
    { id: "POL-004", customer: "Emily Davis", type: "Health", status: "active", premium: "$2,000/yr", startDate: "2024-03-10", endDate: "2025-03-10" },
    { id: "POL-005", customer: "David Wilson", type: "Auto", status: "active", premium: "$1,100/yr", startDate: "2024-04-20", endDate: "2025-04-20" },
  ]);

  const typeColors: { [key: string]: string } = {
    "Auto": "bg-blue-100 text-blue-700",
    "Home": "bg-amber-100 text-amber-700",
    "Life": "bg-red-100 text-red-700",
    "Health": "bg-emerald-100 text-emerald-700",
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Policies</h1>
            <p className="text-slate-600 font-medium mt-1">Manage all insurance policies</p>
          </div>
          <button
            onClick={() => setShowNewPolicyModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + New Policy
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Policies</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">156</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All types</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">148</p>
            <p className="text-xs text-slate-600 font-medium mt-2">94.9% active</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Expiring Soon</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">12</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Next 30 days</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Annual Revenue</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$187K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $1,200</p>
          </div>
        </div>

        {/* Policy Types */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">🚗 Auto Insurance</p>
            <p className="text-2xl font-bold text-blue-700 mt-2">42</p>
            <p className="text-xs text-slate-600 font-medium mt-2">26.9% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">🏠 Home Insurance</p>
            <p className="text-2xl font-bold text-amber-700 mt-2">38</p>
            <p className="text-xs text-slate-600 font-medium mt-2">24.4% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">❤️ Life Insurance</p>
            <p className="text-2xl font-bold text-red-700 mt-2">35</p>
            <p className="text-xs text-slate-600 font-medium mt-2">22.4% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">⚕️ Health Insurance</p>
            <p className="text-2xl font-bold text-emerald-700 mt-2">41</p>
            <p className="text-xs text-slate-600 font-medium mt-2">26.3% of total</p>
          </div>
        </div>

        {/* Policies Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">All Policies</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Policy ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Premium</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Start Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">End Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((policy) => (
                  <tr key={policy.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{policy.id}</td>
                    <td className="p-4 font-medium text-slate-900">{policy.customer}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${typeColors[policy.type]}`}>
                        {policy.type}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        policy.status === "active"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-red-100 text-red-700"
                      }`}>
                        {policy.status === "active" ? "✓ Active" : "✗ Expired"}
                      </span>
                    </td>
                    <td className="p-4 font-bold text-slate-900">{policy.premium}</td>
                    <td className="p-4 text-slate-700">{policy.startDate}</td>
                    <td className="p-4 text-slate-700">{policy.endDate}</td>
                    <td className="p-4">
                      <button
                        onClick={() => setSelectedPolicy(policy)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* New Policy Modal */}
        {showNewPolicyModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowNewPolicyModal(false)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">Create New Policy</h2>
                <button onClick={() => setShowNewPolicyModal(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); alert('Policy created successfully!'); setShowNewPolicyModal(false); }}>
                {/* Customer Information */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Customer Name *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="John Smith" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Customer Email *</label>
                    <input type="email" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="john@example.com" />
                  </div>
                </div>

                {/* Policy Type */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Policy Type *</label>
                  <select required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                    <option value="">Select policy type</option>
                    <option value="auto">Auto Insurance</option>
                    <option value="home">Home Insurance</option>
                    <option value="life">Life Insurance</option>
                    <option value="health">Health Insurance</option>
                  </select>
                </div>

                {/* Coverage Details */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Coverage Amount *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="$500,000" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Premium Amount *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="$1,200/yr" />
                  </div>
                </div>

                {/* Policy Dates */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Start Date *</label>
                    <input type="date" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">End Date *</label>
                    <input type="date" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  </div>
                </div>

                {/* Additional Details */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Policy Details</label>
                  <textarea rows={3} className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="Additional policy information..."></textarea>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 pt-4">
                  <button type="submit" className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95">
                    Create Policy
                  </button>
                  <button type="button" onClick={() => setShowNewPolicyModal(false)} className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all active:scale-95">
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* View Policy Details Modal */}
        {selectedPolicy && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedPolicy(null)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">Policy Details</h2>
                <button onClick={() => setSelectedPolicy(null)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Policy ID</label>
                    <p className="text-lg font-bold text-blue-700">{selectedPolicy.id}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Status</label>
                    <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${
                      selectedPolicy.status === "active"
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-red-100 text-red-700"
                    }`}>
                      {selectedPolicy.status === "active" ? "✓ Active" : "✗ Expired"}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Customer Name</label>
                    <p className="text-lg font-semibold text-slate-900">{selectedPolicy.customer}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Policy Type</label>
                    <p className="text-lg font-semibold text-slate-900">{selectedPolicy.type}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Premium</label>
                    <p className="text-lg font-bold text-slate-900">{selectedPolicy.premium}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Coverage Period</label>
                    <p className="text-lg text-slate-700">{selectedPolicy.startDate} - {selectedPolicy.endDate}</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-200">
                  <div className="flex gap-3">
                    <button
                      onClick={() => setSelectedPolicy(null)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => alert('Edit functionality coming soon!')}
                      className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Edit Policy
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

