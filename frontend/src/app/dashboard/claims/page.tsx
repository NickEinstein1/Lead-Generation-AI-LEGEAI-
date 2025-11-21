"use client";
import { useState, useEffect } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { claimsApi } from "@/lib/api";

export default function ClaimsPage() {
  const [showNewClaimModal, setShowNewClaimModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingClaim, setEditingClaim] = useState<any>(null);
  const [selectedClaim, setSelectedClaim] = useState<any>(null);
  const [claims, setClaims] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const [formData, setFormData] = useState({
    policy_number: "",
    customer_name: "",
    claim_type: "",
    amount: 0,
    status: "pending",
    claim_date: "",
    due_date: "",
    description: ""
  });

  // Fetch claims on mount
  useEffect(() => {
    fetchClaims();
  }, []);

  const fetchClaims = async () => {
    try {
      setLoading(true);
      const data = await claimsApi.getAll();
      setClaims(data);
    } catch (error) {
      console.error("Failed to fetch claims:", error);
      alert("Failed to load claims. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateClaim = async () => {
    if (!formData.policy_number || !formData.customer_name || !formData.claim_type || !formData.amount || !formData.claim_date || !formData.due_date) {
      alert("Please fill in all required fields");
      return;
    }

    try {
      await claimsApi.create(formData);
      await fetchClaims();
      setShowNewClaimModal(false);
      setFormData({ policy_number: "", customer_name: "", claim_type: "", amount: 0, status: "pending", claim_date: "", due_date: "", description: "" });
      alert("Claim created successfully!");
    } catch (error) {
      console.error("Failed to create claim:", error);
      alert("Failed to create claim. Please try again.");
    }
  };

  const handleEditClaim = (claim: any) => {
    setEditingClaim(claim);
    setFormData({
      policy_number: claim.policy_number,
      customer_name: claim.customer_name,
      claim_type: claim.claim_type,
      amount: claim.amount,
      status: claim.status,
      claim_date: claim.claim_date,
      due_date: claim.due_date,
      description: claim.description || ""
    });
    setShowEditModal(true);
  };

  const handleUpdateClaim = async () => {
    if (!formData.policy_number || !formData.customer_name || !formData.claim_type || !formData.amount || !formData.claim_date || !formData.due_date) {
      alert("Please fill in all required fields");
      return;
    }

    try {
      await claimsApi.update(editingClaim.id, formData);
      await fetchClaims();
      setShowEditModal(false);
      setEditingClaim(null);
      setFormData({ policy_number: "", customer_name: "", claim_type: "", amount: 0, status: "pending", claim_date: "", due_date: "", description: "" });
      alert("Claim updated successfully!");
    } catch (error) {
      console.error("Failed to update claim:", error);
      alert("Failed to update claim. Please try again.");
    }
  };

  const handleDeleteClaim = async (claimId: string) => {
    if (confirm("Are you sure you want to delete this claim? This action cannot be undone.")) {
      try {
        await claimsApi.delete(claimId);
        await fetchClaims();
        alert("Claim deleted successfully!");
      } catch (error) {
        console.error("Failed to delete claim:", error);
        alert("Failed to delete claim. Please try again.");
      }
    }
  };

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
          {loading ? (
            <div className="p-8 text-center text-slate-600">Loading claims...</div>
          ) : claims.length === 0 ? (
            <div className="p-8 text-center text-slate-600">No claims found. Create your first claim!</div>
          ) : (
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
                      <td className="p-4 font-bold text-blue-700">{claim.claim_number || claim.id}</td>
                      <td className="p-4 font-medium text-slate-900">{claim.policy_number}</td>
                      <td className="p-4 text-slate-700">{claim.customer_name}</td>
                      <td className="p-4 text-slate-700">{claim.claim_type}</td>
                      <td className="p-4 font-bold text-slate-900">${claim.amount?.toLocaleString() || '0'}</td>
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
                      <td className="p-4 text-slate-700">{claim.claim_date || 'N/A'}</td>
                    <td className="p-4 space-x-2">
                      <button
                        onClick={() => setSelectedClaim(claim)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleEditClaim(claim)}
                        className="text-amber-600 hover:text-amber-800 font-medium text-sm hover:underline"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteClaim(claim.id)}
                        className="text-red-600 hover:text-red-800 font-medium text-sm hover:underline"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
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
                  <label className="block text-sm font-medium text-slate-900 mb-2">Policy Number *</label>
                  <input
                    type="text"
                    value={formData.policy}
                    onChange={(e) => setFormData({...formData, policy: e.target.value})}
                    placeholder="POL-001"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Customer Name *</label>
                  <input
                    type="text"
                    value={formData.customer}
                    onChange={(e) => setFormData({...formData, customer: e.target.value})}
                    placeholder="John Smith"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Claim Type *</label>
                  <select
                    value={formData.type}
                    onChange={(e) => setFormData({...formData, type: e.target.value})}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  >
                    <option value="">Select type...</option>
                    <option value="Auto">🚗 Auto</option>
                    <option value="Home">🏠 Home</option>
                    <option value="Life">❤️ Life</option>
                    <option value="Health">⚕️ Health</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Claim Amount *</label>
                  <input
                    type="text"
                    value={formData.amount}
                    onChange={(e) => setFormData({...formData, amount: e.target.value})}
                    placeholder="$5,000"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Incident Date *</label>
                  <input
                    type="date"
                    value={formData.date}
                    onChange={(e) => setFormData({...formData, date: e.target.value})}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Due Date *</label>
                  <input
                    type="date"
                    value={formData.dueDate}
                    onChange={(e) => setFormData({...formData, dueDate: e.target.value})}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Status</label>
                <select
                  value={formData.status}
                  onChange={(e) => setFormData({...formData, status: e.target.value})}
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                >
                  <option value="pending">⏳ Pending</option>
                  <option value="approved">✓ Approved</option>
                  <option value="rejected">✗ Rejected</option>
                </select>
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
                onClick={handleCreateClaim}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
              >
                Submit Claim
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Claim Modal */}
      {showEditModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowEditModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-2xl font-bold text-slate-900 mb-4">✏️ Edit Claim</h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Policy Number *</label>
                  <input
                    type="text"
                    value={formData.policy}
                    onChange={(e) => setFormData({...formData, policy: e.target.value})}
                    placeholder="POL-001"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Customer Name *</label>
                  <input
                    type="text"
                    value={formData.customer}
                    onChange={(e) => setFormData({...formData, customer: e.target.value})}
                    placeholder="John Smith"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Claim Type *</label>
                  <select
                    value={formData.type}
                    onChange={(e) => setFormData({...formData, type: e.target.value})}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  >
                    <option value="">Select type...</option>
                    <option value="Auto">🚗 Auto</option>
                    <option value="Home">🏠 Home</option>
                    <option value="Life">❤️ Life</option>
                    <option value="Health">⚕️ Health</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Claim Amount *</label>
                  <input
                    type="text"
                    value={formData.amount}
                    onChange={(e) => setFormData({...formData, amount: e.target.value})}
                    placeholder="$5,000"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Incident Date *</label>
                  <input
                    type="date"
                    value={formData.date}
                    onChange={(e) => setFormData({...formData, date: e.target.value})}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Due Date *</label>
                  <input
                    type="date"
                    value={formData.dueDate}
                    onChange={(e) => setFormData({...formData, dueDate: e.target.value})}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Status</label>
                <select
                  value={formData.status}
                  onChange={(e) => setFormData({...formData, status: e.target.value})}
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                >
                  <option value="pending">⏳ Pending</option>
                  <option value="approved">✓ Approved</option>
                  <option value="rejected">✗ Rejected</option>
                </select>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  setShowEditModal(false);
                  setEditingClaim(null);
                  setFormData({ policy: "", customer: "", type: "", amount: "", status: "pending", date: "", dueDate: "" });
                }}
                className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleUpdateClaim}
                className="flex-1 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
              >
                Update Claim
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

