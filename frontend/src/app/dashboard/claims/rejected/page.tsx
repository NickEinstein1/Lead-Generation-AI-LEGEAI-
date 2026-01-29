"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function RejectedClaimsPage() {
  const [showAppealModal, setShowAppealModal] = useState(false);
  const [claims] = useState<any[]>([]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Rejected Claims</h1>
            <p className="text-slate-600 font-medium mt-1">Claims that have been denied or rejected</p>
          </div>
          <button
            onClick={() => setShowAppealModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + Appeal Claim
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Rejected Claims</p>
            <p className="text-3xl font-bold text-red-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No rejection data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Amount</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No amount data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Appeals Filed</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No appeal data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Top Reason</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No reason data</p>
          </div>
        </div>

        {/* Rejected Claims Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Rejected Claims Details</h2>
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
                  <th className="text-left p-4 text-slate-900 font-bold">Rejection Reason</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {claims.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="p-4 text-center text-slate-600 font-medium">No rejected claims available.</td>
                  </tr>
                ) : (
                  claims.map((claim) => (
                    <tr key={claim.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                      <td className="p-4 font-bold text-blue-700">{claim.id}</td>
                      <td className="p-4 font-medium text-slate-900">{claim.policy}</td>
                      <td className="p-4 text-slate-700">{claim.customer}</td>
                      <td className="p-4 text-slate-700">{claim.type}</td>
                      <td className="p-4 font-bold text-slate-900">{claim.amount}</td>
                      <td className="p-4">
                        <span className="px-3 py-1 rounded-full text-xs font-bold bg-red-100 text-red-700">
                          {claim.reason}
                        </span>
                      </td>
                      <td className="p-4 space-x-2">
                        <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Appeal</button>
                        <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Details</button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Appeal Claim Modal */}
      {showAppealModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowAppealModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-2xl font-bold text-slate-900 mb-4">⚖️ Appeal Rejected Claim</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Claim ID</label>
                <input
                  type="text"
                  placeholder="CLM-REJ-001"
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Reason for Appeal</label>
                <textarea
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  rows={5}
                  placeholder="Explain why this claim should be reconsidered..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Additional Evidence</label>
                <div className="border-2 border-dashed border-blue-300 rounded-lg p-6 text-center hover:border-blue-600 transition-colors cursor-pointer">
                  <p className="text-slate-600">📎 Upload supporting documents</p>
                  <p className="text-xs text-slate-500 mt-1">PDF, JPG, PNG up to 10MB</p>
                </div>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAppealModal(false)}
                className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowAppealModal(false);
                  // In a real app, this would submit the appeal
                }}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Submit Appeal
              </button>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}

