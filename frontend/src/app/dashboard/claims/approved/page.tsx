"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";

export default function ApprovedClaimsPage() {
  const router = useRouter();
  const [selectedClaim, setSelectedClaim] = useState<any>(null);
  const [claims] = useState([
    { id: "CLM-APP-001", policy: "POL-001", customer: "John Smith", type: "Auto", amount: "$5,000", approved: "2024-10-15", paidDate: "2024-10-16" },
    { id: "CLM-APP-002", policy: "POL-002", customer: "Sarah Johnson", type: "Home", amount: "$12,500", approved: "2024-10-10", paidDate: "2024-10-12" },
    { id: "CLM-APP-003", policy: "POL-004", customer: "Emily Davis", type: "Health", amount: "$2,300", approved: "2024-10-08", paidDate: "2024-10-09" },
    { id: "CLM-APP-004", policy: "POL-005", customer: "David Wilson", type: "Auto", amount: "$8,750", approved: "2024-10-05", paidDate: "2024-10-07" },
    { id: "CLM-APP-005", policy: "POL-003", customer: "Michael Brown", type: "Life", amount: "$25,000", approved: "2024-10-01", paidDate: "2024-10-03" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Approved Claims</h1>
            <p className="text-slate-600 font-medium mt-1">Claims that have been approved and paid out</p>
          </div>
          <button
            onClick={() => router.push("/dashboard/claims")}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + View Details
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Approved Claims</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">215</p>
            <p className="text-xs text-slate-600 font-medium mt-2">86.7% approval rate</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Payout</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$1.04M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $4,837</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Processing Time</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">3.2 days</p>
            <p className="text-xs text-slate-600 font-medium mt-2">From approval</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Paid This Month</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">$156.8K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">42 claims</p>
          </div>
        </div>

        {/* Approved Claims Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Approved & Paid Claims</h2>
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
                  <th className="text-left p-4 text-slate-900 font-bold">Approved Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Paid Date</th>
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
                    <td className="p-4 text-slate-700">{claim.approved}</td>
                    <td className="p-4 text-slate-700">{claim.paidDate}</td>
                    <td className="p-4 space-x-2">
                      <button
                        onClick={() => setSelectedClaim(claim)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm hover:underline">Receipt</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* View Claim Details Modal */}
        {selectedClaim && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedClaim(null)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">✅ Approved Claim Details</h2>
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
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                      ✓ Approved
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
                    <p className="text-lg font-bold text-emerald-700">{selectedClaim.amount}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Approval Date</label>
                    <p className="text-lg text-slate-700">{selectedClaim.approved}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Payment Date</label>
                    <p className="text-lg text-slate-700">{selectedClaim.paidDate}</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-200">
                  <div className="flex gap-3">
                    <button
                      onClick={() => setSelectedClaim(null)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => alert('Download receipt functionality coming soon!')}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Download Receipt
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

