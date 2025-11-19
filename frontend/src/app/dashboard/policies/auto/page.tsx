"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function AutoInsurancePage() {
  const [showNewPolicyModal, setShowNewPolicyModal] = useState(false);
  const [selectedPolicy, setSelectedPolicy] = useState<any>(null);
  const [policies] = useState([
    { id: "POL-AUTO-001", customer: "John Smith", vehicle: "2022 Toyota Camry", premium: "$1,200/yr", status: "Active", expiry: "2025-01-15" },
    { id: "POL-AUTO-002", customer: "Sarah Johnson", vehicle: "2021 Honda Civic", premium: "$950/yr", status: "Active", expiry: "2025-03-22" },
    { id: "POL-AUTO-003", customer: "Michael Brown", vehicle: "2023 Ford F-150", premium: "$1,450/yr", status: "Active", expiry: "2025-02-10" },
    { id: "POL-AUTO-004", customer: "Emily Davis", vehicle: "2020 Mazda 3", premium: "$875/yr", status: "Expiring Soon", expiry: "2024-11-18" },
    { id: "POL-AUTO-005", customer: "David Wilson", vehicle: "2019 Chevrolet Malibu", premium: "$1,100/yr", status: "Active", expiry: "2025-05-14" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Auto Insurance Policies</h1>
            <p className="text-slate-600 font-medium mt-1">Vehicle insurance coverage and management</p>
          </div>
          <button
            onClick={() => setShowNewPolicyModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + New Auto Policy
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Auto Policies</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">42</p>
            <p className="text-xs text-slate-600 font-medium mt-2">26.9% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Annual Premium</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">$48.6K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $1,157</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">40</p>
            <p className="text-xs text-slate-600 font-medium mt-2">95.2% active</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Expiring Soon</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">2</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Next 30 days</p>
          </div>
        </div>

        {/* Auto Policies Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Auto Insurance Policies</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Policy ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Vehicle</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Premium</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Expiry Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((policy) => (
                  <tr key={policy.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{policy.id}</td>
                    <td className="p-4 font-medium text-slate-900">{policy.customer}</td>
                    <td className="p-4 text-slate-700">{policy.vehicle}</td>
                    <td className="p-4 font-bold text-slate-900">{policy.premium}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        policy.status === "Active"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {policy.status}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{policy.expiry}</td>
                    <td className="p-4 space-x-2">
                      <button
                        onClick={() => setSelectedPolicy(policy)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm hover:underline">Renew</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* New Auto Policy Modal */}
        {showNewPolicyModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowNewPolicyModal(false)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">Create New Auto Insurance Policy</h2>
                <button onClick={() => setShowNewPolicyModal(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); alert('Auto policy created successfully!'); setShowNewPolicyModal(false); }}>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Customer Name *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="John Smith" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Vehicle *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="2022 Toyota Camry" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">VIN Number *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="1HGBH41JXMN109186" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">License Plate *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="ABC-1234" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Coverage Type *</label>
                    <select required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                      <option value="">Select coverage</option>
                      <option value="liability">Liability Only</option>
                      <option value="collision">Collision</option>
                      <option value="comprehensive">Comprehensive</option>
                      <option value="full">Full Coverage</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Premium *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="$1,200/yr" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Start Date *</label>
                    <input type="date" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Expiry Date *</label>
                    <input type="date" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  </div>
                </div>

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
                <h2 className="text-2xl font-bold text-slate-900">🚗 Auto Policy Details</h2>
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
                      selectedPolicy.status === "Active"
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-amber-100 text-amber-700"
                    }`}>
                      {selectedPolicy.status}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Customer Name</label>
                    <p className="text-lg font-semibold text-slate-900">{selectedPolicy.customer}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Vehicle</label>
                    <p className="text-lg font-semibold text-slate-900">{selectedPolicy.vehicle}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Premium</label>
                    <p className="text-lg font-bold text-slate-900">{selectedPolicy.premium}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Expiry Date</label>
                    <p className="text-lg text-slate-700">{selectedPolicy.expiry}</p>
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
                      onClick={() => alert('Renew functionality coming soon!')}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Renew Policy
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

