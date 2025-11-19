"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function ClaimsPage() {
  const [showNewClaimModal, setShowNewClaimModal] = useState(false);
  const [selectedClaim, setSelectedClaim] = useState<any>(null);
  const [claims] = useState([
    { id: "CLM-001", policy: "POL-001", customer: "John Smith", type: "Auto", amount: "$5,000", status: "approved", date: "2024-10-15", dueDate: "2024-11-15" },
    { id: "CLM-002", policy: "POL-002", customer: "Sarah Johnson", type: "Home", amount: "$12,500", status: "pending", date: "2024-10-20", dueDate: "2024-11-20" },
    { id: "CLM-003", policy: "POL-004", customer: "Emily Davis", type: "Health", amount: "$2,300", status: "approved", date: "2024-10-10", dueDate: "2024-11-10" },
    { id: "CLM-004", policy: "POL-005", customer: "David Wilson", type: "Auto", amount: "$8,750", status: "rejected", date: "2024-10-05", dueDate: "2024-11-05" },
    { id: "CLM-005", policy: "POL-001", customer: "John Smith", type: "Auto", amount: "$3,200", status: "pending", date: "2024-10-22", dueDate: "2024-11-22" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Claims</h1>
            <p className="text-slate-600 font-medium mt-1">Manage insurance claims and payouts</p>
          </div>
          <button
            onClick={() => setShowNewClaimModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + New Claim
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Claims</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">248</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All time</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Pending</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">8</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Awaiting review</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Approved</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">215</p>
            <p className="text-xs text-slate-600 font-medium mt-2">86.7% approval rate</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Payout</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$1.2M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $4,839</p>
          </div>
        </div>

        {/* Claims by Status */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">✓ Approved</p>
            <p className="text-2xl font-bold text-emerald-600 mt-2">215</p>
            <p className="text-xs text-slate-600 font-medium mt-2">$1.04M total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">⏳ Pending</p>
            <p className="text-2xl font-bold text-amber-600 mt-2">8</p>
            <p className="text-xs text-slate-600 font-medium mt-2">$31.75K total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">✗ Rejected</p>
            <p className="text-2xl font-bold text-red-600 mt-2">25</p>
            <p className="text-xs text-slate-600 font-medium mt-2">$156.25K total</p>
          </div>
        </div>

        {/* Claims Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Claims</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Claim ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Policy</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Amount</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {claims.map((claim) => (
                  <tr key={claim.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{claim.id}</td>
                    <td className="p-4 font-medium text-slate-900">{claim.policy}</td>
                    <td className="p-4 text-slate-700">{claim.customer}</td>
                    <td className="p-4 text-slate-700">{claim.type}</td>
                    <td className="p-4 font-bold text-slate-900">{claim.amount}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        claim.status === "approved"
                          ? "bg-emerald-100 text-emerald-700"
                          : claim.status === "pending"
                          ? "bg-amber-100 text-amber-700"
                          : "bg-red-100 text-red-700"
                      }`}>
                        {claim.status === "approved" ? "✓ Approved" : claim.status === "pending" ? "⏳ Pending" : "✗ Rejected"}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{claim.date}</td>
                    <td className="p-4">
                      <button
                        onClick={() => setSelectedClaim(claim)}
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
      </div>

      {/* New Claim Modal */}
      {showNewClaimModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowNewClaimModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-2xl font-bold text-slate-900 mb-4">📋 New Claim</h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Policy Number</label>
                  <input
                    type="text"
                    placeholder="POL-001"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Customer Name</label>
                  <input
                    type="text"
                    placeholder="John Smith"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Claim Type</label>
                  <select className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600">
                    <option value="">Select type...</option>
                    <option value="auto">🚗 Auto</option>
                    <option value="home">🏠 Home</option>
                    <option value="life">❤️ Life</option>
                    <option value="health">⚕️ Health</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Claim Amount</label>
                  <input
                    type="text"
                    placeholder="$5,000"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Incident Date</label>
                <input
                  type="date"
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Description</label>
                <textarea
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  rows={4}
                  placeholder="Describe the incident and claim details..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Supporting Documents</label>
                <div className="border-2 border-dashed border-blue-300 rounded-lg p-6 text-center hover:border-blue-600 transition-colors cursor-pointer">
                  <p className="text-slate-600">📎 Click to upload or drag and drop</p>
                  <p className="text-xs text-slate-500 mt-1">PDF, JPG, PNG up to 10MB</p>
                </div>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowNewClaimModal(false)}
                className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowNewClaimModal(false);
                  // In a real app, this would submit the claim
                }}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Submit Claim
              </button>
            </div>
          </div>
        </div>
      )}

      {/* View Claim Details Modal */}
      {selectedClaim && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedClaim(null)}>
          <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-slate-900">📋 Claim Details</h2>
              <button onClick={() => setSelectedClaim(null)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Claim ID</label>
                  <p className="text-lg font-bold text-blue-700">{selectedClaim.id}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Status</label>
                  <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${
                    selectedClaim.status === "approved"
                      ? "bg-emerald-100 text-emerald-700"
                      : selectedClaim.status === "pending"
                      ? "bg-amber-100 text-amber-700"
                      : "bg-red-100 text-red-700"
                  }`}>
                    {selectedClaim.status === "approved" ? "✓ Approved" : selectedClaim.status === "pending" ? "⏳ Pending" : "✗ Rejected"}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Customer Name</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedClaim.customer}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Policy Number</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedClaim.policy}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Claim Type</label>
                  <p className="text-lg text-slate-700">{selectedClaim.type}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Claim Amount</label>
                  <p className="text-lg font-bold text-slate-900">{selectedClaim.amount}</p>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-500 mb-1">Submission Date</label>
                <p className="text-lg text-slate-700">{selectedClaim.date}</p>
              </div>

              <div className="pt-4 border-t border-slate-200">
                <div className="flex gap-3">
                  <button
                    onClick={() => setSelectedClaim(null)}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Close
                  </button>
                  {selectedClaim.status === "pending" && (
                    <button
                      onClick={() => alert('Approve/Reject functionality coming soon!')}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Process Claim
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}

