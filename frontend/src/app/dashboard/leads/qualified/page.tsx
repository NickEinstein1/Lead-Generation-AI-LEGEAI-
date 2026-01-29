"use client";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import { listLeads } from "@/lib/api";

export default function QualifiedLeadsPage() {
  const router = useRouter();
  const [selectedLead, setSelectedLead] = useState<any>(null);
  const [leads, setLeads] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const res = await listLeads(50, 0);
        setLeads(res.items || []);
      } catch (e: any) {
        setError(e?.message || "Failed to load leads");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const stats = useMemo(() => {
    const total = leads.length;
    const scores = leads
      .map((lead) => lead.attributes?.score)
      .filter((score: any) => typeof score === "number");
    const avgScore = scores.length > 0 ? (scores.reduce((sum: number, s: number) => sum + s, 0) / scores.length) : null;
    return { total, avgScore };
  }, [leads]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Qualified Leads</h1>
            <p className="text-slate-600 font-medium mt-1">High-potential leads ready for conversion</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Qualified Lead
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Qualified</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">{loading ? "-" : stats.total}</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Based on available leads</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Hot Leads</p>
            <p className="text-3xl font-bold text-red-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No status data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Lead Score</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">{stats.avgScore ? stats.avgScore.toFixed(1) : "-"}</p>
            <p className="text-xs text-slate-600 font-medium mt-2">From scored leads</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Pipeline Value</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No value data</p>
          </div>
        </div>

        {/* Qualified Leads Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Qualified Leads Pipeline</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Lead ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Phone</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Score</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Value</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={8} className="p-4 text-center text-slate-600 font-medium">Loading leads...</td>
                  </tr>
                ) : error ? (
                  <tr>
                    <td colSpan={8} className="p-4 text-center text-red-600 font-medium">{error}</td>
                  </tr>
                ) : leads.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="p-4 text-center text-slate-600 font-medium">No leads available.</td>
                  </tr>
                ) : (
                  leads.map((lead) => {
                    const contact = lead.contact || lead.contact_info || {};
                    const name = [contact.first_name, contact.last_name].filter(Boolean).join(" ") || "-";
                    const score = typeof lead.attributes?.score === "number" ? lead.attributes.score : null;
                    const status = lead.status || "Unknown";
                    const statusClass = status === "Hot"
                      ? "bg-red-100 text-red-700"
                      : status === "Warm"
                        ? "bg-amber-100 text-amber-700"
                        : "bg-slate-100 text-slate-700";

                    return (
                      <tr key={lead.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                        <td className="p-4 font-bold text-blue-700">{lead.id}</td>
                        <td className="p-4 font-medium text-slate-900">{name}</td>
                        <td className="p-4 text-slate-700">{contact.email || "-"}</td>
                        <td className="p-4 text-slate-700">{contact.phone || "-"}</td>
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-slate-200 rounded-full h-2">
                              <div className="bg-emerald-600 h-2 rounded-full" style={{ width: `${score ?? 0}%` }}></div>
                            </div>
                            <span className="font-bold text-slate-900">{score ?? "-"}</span>
                          </div>
                        </td>
                        <td className="p-4">
                          <span className={`px-3 py-1 rounded-full text-xs font-bold ${statusClass}`}>
                            {status}
                          </span>
                        </td>
                        <td className="p-4 font-bold text-slate-900">{lead.attributes?.estimated_value ?? "-"}</td>
                        <td className="p-4 space-x-2">
                          <button
                            onClick={() => setSelectedLead(lead)}
                            className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                          >
                            View
                          </button>
                          <button className="text-slate-600 hover:text-slate-800 font-medium text-sm hover:underline">Convert</button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* View Lead Details Modal */}
        {selectedLead && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedLead(null)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">🎯 Qualified Lead Details</h2>
                <button onClick={() => setSelectedLead(null)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Lead ID</label>
                    <p className="text-lg font-bold text-blue-700">{selectedLead.id}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Score</label>
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                      {selectedLead.attributes?.score ?? "-"}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Lead Name</label>
                    <p className="text-lg font-semibold text-slate-900">
                      {[selectedLead.contact?.first_name, selectedLead.contact?.last_name].filter(Boolean).join(" ") || "-"}
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Email Address</label>
                    <p className="text-lg text-slate-700">{selectedLead.contact?.email || "-"}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Phone Number</label>
                    <p className="text-lg text-slate-700">{selectedLead.contact?.phone || "-"}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Interest</label>
                    <p className="text-lg text-slate-700">{selectedLead.product_interest || "-"}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Estimated Value</label>
                    <p className="text-lg font-bold text-slate-900">{selectedLead.attributes?.estimated_value ?? "-"}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Source</label>
                    <p className="text-lg text-slate-700">{selectedLead.source || "-"}</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-200">
                  <div className="flex gap-3">
                    <button
                      onClick={() => setSelectedLead(null)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => router.push(`/dashboard/leads/${selectedLead.id}`)}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      View Full Details
                    </button>
                    <button
                      onClick={() => alert('Convert to customer functionality coming soon!')}
                      className="flex-1 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Convert
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

