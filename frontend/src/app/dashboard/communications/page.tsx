"use client";
import { useState, useEffect } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { communicationsApi } from "@/lib/api";

export default function CommunicationsPage() {
  const [showNewCommModal, setShowNewCommModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingComm, setEditingComm] = useState<any>(null);
  const [selectedComm, setSelectedComm] = useState<any>(null);
  const [communications, setCommunications] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const [formData, setFormData] = useState({
    comm_type: "email",
    customer_name: "",
    subject: "",
    status: "sent",
    comm_date: "",
    channel: "Email",
    content: ""
  });

  // Fetch communications on mount
  useEffect(() => {
    fetchCommunications();
  }, []);

  const fetchCommunications = async () => {
    try {
      setLoading(true);
      const data = await communicationsApi.getAll();
      setCommunications(data);
    } catch (error) {
      console.error("Failed to fetch communications:", error);
      alert("Failed to load communications. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateCommunication = async () => {
    if (!formData.customer_name || !formData.subject || !formData.comm_date) {
      alert("Please fill in all required fields");
      return;
    }

    try {
      await communicationsApi.create(formData);
      await fetchCommunications();
      setShowNewCommModal(false);
      setFormData({ comm_type: "email", customer_name: "", subject: "", status: "sent", comm_date: "", channel: "Email", content: "" });
      alert("Communication created successfully!");
    } catch (error) {
      console.error("Failed to create communication:", error);
      alert("Failed to create communication. Please try again.");
    }
  };

  const handleEditCommunication = (comm: any) => {
    setEditingComm(comm);
    setFormData({
      comm_type: comm.comm_type,
      customer_name: comm.customer_name,
      subject: comm.subject,
      status: comm.status,
      comm_date: comm.comm_date,
      channel: comm.channel,
      content: comm.content || ""
    });
    setShowEditModal(true);
  };

  const handleUpdateCommunication = async () => {
    if (!formData.customer_name || !formData.subject || !formData.comm_date) {
      alert("Please fill in all required fields");
      return;
    }

    try {
      await communicationsApi.update(editingComm.id, formData);
      await fetchCommunications();
      setShowEditModal(false);
      setEditingComm(null);
      setFormData({ comm_type: "email", customer_name: "", subject: "", status: "sent", comm_date: "", channel: "Email", content: "" });
      alert("Communication updated successfully!");
    } catch (error) {
      console.error("Failed to update communication:", error);
      alert("Failed to update communication. Please try again.");
    }
  };

  const handleDeleteCommunication = async (commId: string) => {
    if (confirm("Are you sure you want to delete this communication? This action cannot be undone.")) {
      try {
        await communicationsApi.delete(commId);
        await fetchCommunications();
        alert("Communication deleted successfully!");
      } catch (error) {
        console.error("Failed to delete communication:", error);
        alert("Failed to delete communication. Please try again.");
      }
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Communications</h1>
            <p className="text-slate-600 font-medium mt-1">Manage customer communications and campaigns</p>
          </div>
          <button
            onClick={() => setShowNewCommModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + New Communication
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Sent</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">3,456</p>
            <p className="text-xs text-slate-600 font-medium mt-2">This month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Delivery Rate</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">98.5%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Successfully delivered</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Open Rate</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">42.3%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Email opens</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Click Rate</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">18.7%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Link clicks</p>
          </div>
        </div>

        {/* Communication Channels */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">📧 Emails</p>
            <p className="text-2xl font-bold text-blue-700 mt-2">1,234</p>
            <p className="text-xs text-slate-600 font-medium mt-2">35.7% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">💬 SMS</p>
            <p className="text-2xl font-bold text-emerald-700 mt-2">1,567</p>
            <p className="text-xs text-slate-600 font-medium mt-2">45.3% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">☎️ Calls</p>
            <p className="text-2xl font-bold text-red-700 mt-2">456</p>
            <p className="text-xs text-slate-600 font-medium mt-2">13.2% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">📢 Campaigns</p>
            <p className="text-2xl font-bold text-purple-700 mt-2">45</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Active campaigns</p>
          </div>
        </div>

        {/* Communications Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Communications</h2>
          </div>
          {loading ? (
            <div className="p-8 text-center text-slate-600">Loading communications...</div>
          ) : communications.length === 0 ? (
            <div className="p-8 text-center text-slate-600">No communications found. Create your first communication!</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-blue-50 border-b-2 border-blue-200">
                  <tr>
                    <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Subject</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Channel</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Date</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {communications.map((comm) => (
                    <tr key={comm.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                      <td className="p-4">
                        <span className="text-lg">
                          {comm.comm_type === "email" ? "📧" : comm.comm_type === "sms" ? "💬" : "☎️"}
                        </span>
                      </td>
                      <td className="p-4 font-medium text-slate-900">{comm.customer_name}</td>
                      <td className="p-4 text-slate-700">{comm.subject}</td>
                      <td className="p-4 text-slate-700">{comm.channel}</td>
                      <td className="p-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                          comm.status === "sent" || comm.status === "delivered" || comm.status === "completed"
                            ? "bg-emerald-100 text-emerald-700"
                            : "bg-amber-100 text-amber-700"
                        }`}>
                          {comm.status === "sent" ? "✓ Sent" : comm.status === "delivered" ? "✓ Delivered" : "✓ Completed"}
                        </span>
                      </td>
                      <td className="p-4 text-slate-700">{comm.comm_date || 'N/A'}</td>
                    <td className="p-4 space-x-2">
                      <button
                        onClick={() => setSelectedComm(comm)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleEditCommunication(comm)}
                        className="text-amber-600 hover:text-amber-800 font-medium text-sm hover:underline"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteCommunication(comm.id)}
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

        {/* New Communication Modal */}
        {showNewCommModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowNewCommModal(false)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">💬 New Communication</h2>
                <button onClick={() => setShowNewCommModal(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Communication Type *</label>
                  <select
                    value={formData.type}
                    onChange={(e) => {
                      const type = e.target.value;
                      const channelMap: any = { email: "Email", sms: "SMS", call: "Phone" };
                      setFormData({...formData, type, channel: channelMap[type]});
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="email">📧 Email</option>
                    <option value="sms">💬 SMS</option>
                    <option value="call">☎️ Phone Call</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Customer Name *</label>
                  <input
                    type="text"
                    value={formData.customer}
                    onChange={(e) => setFormData({...formData, customer: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="John Smith"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Subject/Topic *</label>
                  <input
                    type="text"
                    value={formData.subject}
                    onChange={(e) => setFormData({...formData, subject: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Policy Renewal Reminder"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Status</label>
                  <select
                    value={formData.status}
                    onChange={(e) => setFormData({...formData, status: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="sent">✓ Sent</option>
                    <option value="delivered">✓ Delivered</option>
                    <option value="completed">✓ Completed</option>
                    <option value="pending">⏳ Pending</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Date *</label>
                  <input
                    type="date"
                    value={formData.date}
                    onChange={(e) => setFormData({...formData, date: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div className="flex gap-3 pt-4">
                  <button
                    onClick={() => {
                      setShowNewCommModal(false);
                      setFormData({ type: "email", customer: "", subject: "", status: "sent", date: "", channel: "Email" });
                    }}
                    className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreateCommunication}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Create Communication
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Edit Communication Modal */}
        {showEditModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowEditModal(false)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">✏️ Edit Communication</h2>
                <button onClick={() => setShowEditModal(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Communication Type *</label>
                  <select
                    value={formData.type}
                    onChange={(e) => {
                      const type = e.target.value;
                      const channelMap: any = { email: "Email", sms: "SMS", call: "Phone" };
                      setFormData({...formData, type, channel: channelMap[type]});
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="email">📧 Email</option>
                    <option value="sms">💬 SMS</option>
                    <option value="call">☎️ Phone Call</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Customer Name *</label>
                  <input
                    type="text"
                    value={formData.customer}
                    onChange={(e) => setFormData({...formData, customer: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="John Smith"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Subject/Topic *</label>
                  <input
                    type="text"
                    value={formData.subject}
                    onChange={(e) => setFormData({...formData, subject: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Policy Renewal Reminder"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Status</label>
                  <select
                    value={formData.status}
                    onChange={(e) => setFormData({...formData, status: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="sent">✓ Sent</option>
                    <option value="delivered">✓ Delivered</option>
                    <option value="completed">✓ Completed</option>
                    <option value="pending">⏳ Pending</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Date *</label>
                  <input
                    type="date"
                    value={formData.date}
                    onChange={(e) => setFormData({...formData, date: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div className="flex gap-3 pt-4">
                  <button
                    onClick={() => {
                      setShowEditModal(false);
                      setEditingComm(null);
                      setFormData({ type: "email", customer: "", subject: "", status: "sent", date: "", channel: "Email" });
                    }}
                    className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleUpdateCommunication}
                    className="flex-1 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Update Communication
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* View Communication Details Modal */}
        {selectedComm && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedComm(null)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">💬 Communication Details</h2>
                <button onClick={() => setSelectedComm(null)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Type</label>
                  <p className="text-lg font-semibold text-slate-900">
                    {selectedComm.type === "email" ? "📧 Email" : selectedComm.type === "sms" ? "💬 SMS" : "☎️ Phone Call"}
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Customer</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedComm.customer}</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Subject/Topic</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedComm.subject}</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Channel</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedComm.channel}</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Status</label>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                    selectedComm.status === "sent" || selectedComm.status === "delivered" || selectedComm.status === "completed"
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-amber-100 text-amber-700"
                  }`}>
                    {selectedComm.status === "sent" ? "✓ Sent" : selectedComm.status === "delivered" ? "✓ Delivered" : "✓ Completed"}
                  </span>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Date</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedComm.date}</p>
                </div>

                <div className="flex gap-3 pt-4">
                  <button
                    onClick={() => setSelectedComm(null)}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

