"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";

export default function DocumentsPage() {
  const router = useRouter();
  const [selectedDocument, setSelectedDocument] = useState<any>(null);
  const [documents] = useState([
    { id: 1, title: "Auto Insurance Agreement", customer: "John Smith", status: "signed", created: "2024-10-15", signed: "2024-10-16", type: "Agreement" },
    { id: 2, title: "Home Insurance Policy", customer: "Sarah Johnson", status: "pending", created: "2024-10-20", signed: null, type: "Policy" },
    { id: 3, title: "Life Insurance Rider", customer: "Michael Brown", status: "signed", created: "2024-10-10", signed: "2024-10-12", type: "Rider" },
    { id: 4, title: "Health Insurance Enrollment", customer: "Emily Davis", status: "declined", created: "2024-10-18", signed: null, type: "Enrollment" },
    { id: 5, title: "Claim Form - Auto", customer: "David Wilson", status: "pending", created: "2024-10-22", signed: null, type: "Claim Form" },
  ]);

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
            onClick={() => router.push("/dashboard/file-library")}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + New Document
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Documents</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">1,245</p>
            <p className="text-xs text-slate-600 font-medium mt-2">All time</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Signed</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">1,180</p>
            <p className="text-xs text-slate-600 font-medium mt-2">94.8% signed</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Pending Signature</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">42</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Awaiting signature</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Signing Time</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">2.3 days</p>
            <p className="text-xs text-slate-600 font-medium mt-2">From creation</p>
          </div>
        </div>

        {/* Document Types */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">📋 Agreements</p>
            <p className="text-2xl font-bold text-blue-700 mt-2">456</p>
            <p className="text-xs text-slate-600 font-medium mt-2">36.6% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">📄 Policies</p>
            <p className="text-2xl font-bold text-amber-700 mt-2">389</p>
            <p className="text-xs text-slate-600 font-medium mt-2">31.2% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">📝 Forms</p>
            <p className="text-2xl font-bold text-red-700 mt-2">267</p>
            <p className="text-xs text-slate-600 font-medium mt-2">21.4% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">🔗 Other</p>
            <p className="text-2xl font-bold text-emerald-700 mt-2">133</p>
            <p className="text-xs text-slate-600 font-medium mt-2">10.7% of total</p>
          </div>
        </div>

        {/* Documents Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Documents</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Title</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Created</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Signed</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-medium text-slate-900">{doc.title}</td>
                    <td className="p-4 text-slate-700">{doc.customer}</td>
                    <td className="p-4 text-slate-700">{doc.type}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        doc.status === "signed"
                          ? "bg-emerald-100 text-emerald-700"
                          : doc.status === "pending"
                          ? "bg-amber-100 text-amber-700"
                          : "bg-red-100 text-red-700"
                      }`}>
                        {doc.status === "signed" ? "✓ Signed" : doc.status === "pending" ? "✍️ Pending" : "✗ Declined"}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{doc.created}</td>
                    <td className="p-4 text-slate-700">{doc.signed || "-"}</td>
                    <td className="p-4">
                      <button
                        onClick={() => setSelectedDocument(doc)}
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

