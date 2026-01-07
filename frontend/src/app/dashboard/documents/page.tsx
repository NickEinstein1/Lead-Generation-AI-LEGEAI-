"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import { fileDocumentsApi } from "@/lib/api";

	export default function DocumentsPage() {
	  const router = useRouter();
	  const [showNewDocModal, setShowNewDocModal] = useState(false);
	  const [showEditModal, setShowEditModal] = useState(false);
	  const [editingDocument, setEditingDocument] = useState<any>(null);
	  const [selectedDocument, setSelectedDocument] = useState<any>(null);
	  const [documents, setDocuments] = useState<any[]>([]);
	  const [loading, setLoading] = useState(true);
	  const [selectedFile, setSelectedFile] = useState<File | null>(null);

	  // Aggregated stats for dashboard cards
	  const [stats, setStats] = useState<any | null>(null);

  const [formData, setFormData] = useState({
    title: "",
    category: "policy",
    description: "",
    status: "active"
  });

	  // Fetch documents and stats on mount
	  useEffect(() => {
	    fetchDocuments();
	    fetchDocumentStats();
	  }, []);

	  const fetchDocuments = async () => {
	    try {
	      setLoading(true);
	      const response = await fileDocumentsApi.getAll({ status: 'active' });
	      setDocuments(response.documents || []);
	    } catch (error) {
	      console.error("Failed to fetch documents:", error);
	      alert("Failed to load documents. Please try again.");
	    } finally {
	      setLoading(false);
	    }
	  };

	  const fetchDocumentStats = async () => {
	    try {
	      const data = await fileDocumentsApi.getStats();
	      setStats(data);
	    } catch (error) {
	      console.error("Failed to fetch document stats:", error);
	    }
	  };

  const handleCreateDocument = async () => {
    if (!formData.title || !formData.category || !selectedFile) {
      alert("Please fill in all required fields and select a file");
      return;
    }

    try {
      await fileDocumentsApi.upload(selectedFile, formData.title, formData.category, formData.description);
      await fetchDocuments();
      setShowNewDocModal(false);
      setFormData({ title: "", category: "policy", description: "", status: "active" });
      setSelectedFile(null);
      alert("Document uploaded successfully!");
    } catch (error) {
      console.error("Failed to upload document:", error);
      alert("Failed to upload document. Please try again.");
    }
  };

  const handleEditDocument = (document: any) => {
    setEditingDocument(document);
    setFormData({
      title: document.title,
      category: document.category,
      description: document.description || "",
      status: document.status
    });
    setShowEditModal(true);
  };

  const handleUpdateDocument = async () => {
    // Note: The current API doesn't support updating document metadata
    // This would need to be added to the backend API
    alert("Document update functionality will be available soon!");
    setShowEditModal(false);
    setEditingDocument(null);
    setFormData({ title: "", category: "policy", description: "", status: "active" });
  };

  const handleDeleteDocument = async (documentId: number) => {
    if (confirm("Are you sure you want to delete this document? This action cannot be undone.")) {
      try {
        await fileDocumentsApi.delete(documentId, false);
        await fetchDocuments();
        alert("Document deleted successfully!");
      } catch (error) {
        console.error("Failed to delete document:", error);
        alert("Failed to delete document. Please try again.");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Documents</h1>
            <p className="text-slate-600 font-medium mt-1">Manage documents and e-signatures</p>
          </div>
          <button
            onClick={() => setShowNewDocModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + New Document
          </button>
        </div>

	        {/* Stats */}
	        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
	          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
	            <p className="text-slate-600 text-sm font-medium">Total Documents</p>
	            <p className="text-3xl font-bold text-blue-700 mt-2">
	              {stats ? stats.total_documents.toLocaleString() : documents.length.toLocaleString()}
	            </p>
	            <p className="text-xs text-slate-600 font-medium mt-2">Across all types</p>
	          </div>
	          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
	            <p className="text-slate-600 text-sm font-medium">Signed</p>
	            <p className="text-3xl font-bold text-emerald-600 mt-2">
	              {stats ? stats.signed_documents.toLocaleString() : "-"}
	            </p>
	            <p className="text-xs text-slate-600 font-medium mt-2">
	              {stats && stats.total_documents
	                ? `${Math.round((stats.signed_documents / Math.max(1, stats.total_documents)) * 100)}% of docs`
	                : "Signed documents"}
	            </p>
	          </div>
	          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
	            <p className="text-slate-600 text-sm font-medium">Pending Signature</p>
	            <p className="text-3xl font-bold text-amber-600 mt-2">
	              {stats ? stats.pending_signature_documents.toLocaleString() : "-"}
	            </p>
	            <p className="text-xs text-slate-600 font-medium mt-2">Awaiting completion</p>
	          </div>
	          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
	            <p className="text-slate-600 text-sm font-medium">Recent Uploads (7d)</p>
	            <p className="text-3xl font-bold text-purple-600 mt-2">
	              {stats ? stats.recent_uploads_last_7_days.toLocaleString() : "-"}
	            </p>
	            <p className="text-xs text-slate-600 font-medium mt-2">Last 7 days</p>
	          </div>
	        </div>

	        {/* Document Types */}
	        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
	          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
	            <p className="text-slate-600 text-sm font-medium">📋 Agreements</p>
	            <p className="text-2xl font-bold text-blue-700 mt-2">
	              {stats ? (stats.by_category?.agreement || 0).toLocaleString() : "-"}
	            </p>
	            <p className="text-xs text-slate-600 font-medium mt-2">By category</p>
	          </div>
	          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
	            <p className="text-slate-600 text-sm font-medium">📄 Policies</p>
	            <p className="text-2xl font-bold text-amber-700 mt-2">
	              {stats ? (stats.by_category?.policy || 0).toLocaleString() : "-"}
	            </p>
	            <p className="text-xs text-slate-600 font-medium mt-2">By category</p>
	          </div>
	          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
	            <p className="text-slate-600 text-sm font-medium">📝 Forms</p>
	            <p className="text-2xl font-bold text-red-700 mt-2">
	              {stats ? (stats.by_category?.form || 0).toLocaleString() : "-"}
	            </p>
	            <p className="text-xs text-slate-600 font-medium mt-2">By category</p>
	          </div>
	          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
	            <p className="text-slate-600 text-sm font-medium">🔗 Other</p>
	            <p className="text-2xl font-bold text-emerald-700 mt-2">
	              {stats ? (stats.by_category?.other || 0).toLocaleString() : "-"}
	            </p>
	            <p className="text-xs text-slate-600 font-medium mt-2">By category</p>
	          </div>
	        </div>

        {/* Documents Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Documents</h2>
          </div>
          {loading ? (
            <div className="p-8 text-center text-slate-600">Loading documents...</div>
          ) : documents.length === 0 ? (
            <div className="p-8 text-center text-slate-600">No documents found. Upload your first document!</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-blue-50 border-b-2 border-blue-200">
                  <tr>
                    <th className="text-left p-4 text-slate-900 font-bold">Title</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Filename</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Category</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Size</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Created</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((doc) => (
                    <tr key={doc.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                      <td className="p-4 font-medium text-slate-900">{doc.title}</td>
                      <td className="p-4 text-slate-700">{doc.original_filename}</td>
                      <td className="p-4 text-slate-700 capitalize">{doc.category}</td>
                      <td className="p-4 text-slate-700 uppercase">{doc.file_type}</td>
                      <td className="p-4 text-slate-700">{(doc.file_size / 1024).toFixed(1)} KB</td>
                      <td className="p-4 text-slate-700">{doc.created_at ? new Date(doc.created_at).toLocaleDateString() : 'N/A'}</td>
                      <td className="p-4 space-x-2">
                        <a
                          href={fileDocumentsApi.download(doc.id)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                        >
                          Download
                        </a>
                        <button
                          onClick={() => setSelectedDocument(doc)}
                          className="text-green-600 hover:text-green-800 font-medium text-sm hover:underline"
                        >
                          View
                        </button>
                        <button
                          onClick={() => handleDeleteDocument(doc.id)}
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

        {/* New Document Modal */}
        {showNewDocModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowNewDocModal(false)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">📄 New Document</h2>
                <button onClick={() => setShowNewDocModal(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Document Title *</label>
                  <input
                    type="text"
                    value={formData.title}
                    onChange={(e) => setFormData({...formData, title: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Auto Insurance Agreement"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">File Upload *</label>
                  <input
                    type="file"
                    onChange={handleFileChange}
                    accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.txt,.png,.jpg,.jpeg"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  {selectedFile && (
                    <p className="text-sm text-slate-600 mt-1">Selected: {selectedFile.name}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Category *</label>
                  <select
                    value={formData.category}
                    onChange={(e) => setFormData({...formData, category: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="policy">Policy</option>
                    <option value="claim">Claim</option>
                    <option value="agreement">Agreement</option>
                    <option value="enrollment">Enrollment</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Description</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({...formData, description: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    rows={3}
                    placeholder="Optional description..."
                  />
                </div>

                <div className="flex gap-3 pt-4">
                  <button
                    onClick={() => {
                      setShowNewDocModal(false);
                      setFormData({ title: "", customer: "", type: "", status: "pending", created: "", signed: "" });
                    }}
                    className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreateDocument}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Create Document
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Edit Document Modal */}
        {showEditModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowEditModal(false)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">✏️ Edit Document</h2>
                <button onClick={() => setShowEditModal(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Document Title *</label>
                  <input
                    type="text"
                    value={formData.title}
                    onChange={(e) => setFormData({...formData, title: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Auto Insurance Agreement"
                  />
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
                  <label className="block text-sm font-medium text-slate-700 mb-1">Document Type *</label>
                  <select
                    value={formData.type}
                    onChange={(e) => setFormData({...formData, type: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select type</option>
                    <option value="Agreement">Agreement</option>
                    <option value="Policy">Policy</option>
                    <option value="Rider">Rider</option>
                    <option value="Enrollment">Enrollment</option>
                    <option value="Claim Form">Claim Form</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Status</label>
                  <select
                    value={formData.status}
                    onChange={(e) => setFormData({...formData, status: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="pending">✍️ Pending</option>
                    <option value="signed">✓ Signed</option>
                    <option value="declined">✗ Declined</option>
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Created Date *</label>
                    <input
                      type="date"
                      value={formData.created}
                      onChange={(e) => setFormData({...formData, created: e.target.value})}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Signed Date</label>
                    <input
                      type="date"
                      value={formData.signed}
                      onChange={(e) => setFormData({...formData, signed: e.target.value})}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>

                <div className="flex gap-3 pt-4">
                  <button
                    onClick={() => {
                      setShowEditModal(false);
                      setEditingDocument(null);
                      setFormData({ title: "", customer: "", type: "", status: "pending", created: "", signed: "" });
                    }}
                    className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleUpdateDocument}
                    className="flex-1 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    Update Document
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* View Document Details Modal */}
        {selectedDocument && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedDocument(null)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">📄 Document Details</h2>
                <button onClick={() => setSelectedDocument(null)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Document ID</label>
                    <p className="text-lg font-bold text-blue-700">DOC-{selectedDocument.id}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Status</label>
                    <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${
                      selectedDocument.status === "signed"
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-amber-100 text-amber-700"
                    }`}>
                      {selectedDocument.status === "signed" ? "✓ Signed" : "⏳ Pending"}
                    </span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Document Title</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedDocument.title}</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Customer Name</label>
                    <p className="text-lg text-slate-700">{selectedDocument.customer}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Document Type</label>
                    <p className="text-lg text-slate-700">{selectedDocument.type}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Created Date</label>
                    <p className="text-lg text-slate-700">{selectedDocument.created}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Signed Date</label>
                    <p className="text-lg text-slate-700">{selectedDocument.signed || "Not signed yet"}</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-200">
                  <div className="flex gap-3">
                    <button
                      onClick={() => setSelectedDocument(null)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => alert('Download document functionality coming soon!')}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Download
                    </button>
                    {selectedDocument.status === "pending" && (
                      <button
                        onClick={() => alert('Send reminder functionality coming soon!')}
                        className="flex-1 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                      >
                        Send Reminder
                      </button>
                    )}
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

