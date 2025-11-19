"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function LifeInsurancePage() {
  const [showNewPolicyModal, setShowNewPolicyModal] = useState(false);
  const [selectedPolicy, setSelectedPolicy] = useState<any>(null);
  const [policies] = useState([
    { id: "POL-LIFE-001", customer: "John Smith", coverage: "$500,000", premium: "$45/mo", status: "Active", expiry: "2045-01-15" },
    { id: "POL-LIFE-002", customer: "Sarah Johnson", coverage: "$750,000", premium: "$68/mo", status: "Active", expiry: "2050-03-22" },
    { id: "POL-LIFE-003", customer: "Michael Brown", coverage: "$1,000,000", premium: "$92/mo", status: "Active", expiry: "2048-02-10" },
    { id: "POL-LIFE-004", customer: "Emily Davis", coverage: "$250,000", premium: "$28/mo", status: "Active", expiry: "2055-04-18" },
    { id: "POL-LIFE-005", customer: "David Wilson", coverage: "$600,000", premium: "$55/mo", status: "Active", expiry: "2046-11-14" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Life Insurance Policies</h1>
            <p className="text-slate-600 font-medium mt-1">Life insurance and protection coverage</p>
          </div>
          <button
            onClick={() => setShowNewPolicyModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + New Life Policy
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Life Policies</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">35</p>
            <p className="text-xs text-slate-600 font-medium mt-2">22.4% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Coverage</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">$18.5M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $528K</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Monthly Premium</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$2,145</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $61/mo</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">35</p>
            <p className="text-xs text-slate-600 font-medium mt-2">100% active</p>
          </div>
        </div>

        {/* Life Policies Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Life Insurance Policies</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Policy ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Coverage Amount</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Premium</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Maturity Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((policy) => (
                  <tr key={policy.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{policy.id}</td>
                    <td className="p-4 font-medium text-slate-900">{policy.customer}</td>
                    <td className="p-4 font-bold text-slate-900">{policy.coverage}</td>
                    <td className="p-4 font-bold text-slate-900">{policy.premium}</td>
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
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
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm hover:underline">Modify</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* New Life Policy Modal */}
        {showNewPolicyModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowNewPolicyModal(false)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">Create New Life Insurance Policy</h2>
                <button onClick={() => setShowNewPolicyModal(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); alert('Life policy created successfully!'); setShowNewPolicyModal(false); }}>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Customer Name *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="John Smith" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Date of Birth *</label>
                    <input type="date" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Policy Type *</label>
                    <select required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                      <option value="">Select type</option>
                      <option value="term">Term Life</option>
                      <option value="whole">Whole Life</option>
                      <option value="universal">Universal Life</option>
                      <option value="variable">Variable Life</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Coverage Amount *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="$500,000" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Premium *</label>
                    <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="$45/mo" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Term Length *</label>
                    <select required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                      <option value="">Select term</option>
                      <option value="10">10 Years</option>
                      <option value="20">20 Years</option>
                      <option value="30">30 Years</option>
                      <option value="lifetime">Lifetime</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Beneficiaries *</label>
                  <input type="text" required className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="Spouse, Children" />
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
                <h2 className="text-2xl font-bold text-slate-900">💼 Life Policy Details</h2>
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
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
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
                    <label className="block text-sm font-medium text-slate-500 mb-1">Coverage</label>
                    <p className="text-lg font-bold text-slate-900">{selectedPolicy.coverage}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Premium</label>
                    <p className="text-lg font-bold text-slate-900">{selectedPolicy.premium}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Maturity Date</label>
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
                      onClick={() => alert('Modify functionality coming soon!')}
                      className="flex-1 bg-slate-600 hover:bg-slate-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Modify Policy
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

