"use client";
import { useEffect, useMemo, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { listLeads } from "@/lib/api";

export default function NewLeadsPage() {
  const [showAddLeadModal, setShowAddLeadModal] = useState(false);
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

  const parseValue = (value: unknown) => {
    if (typeof value === "number") return value;
    if (typeof value !== "string") return null;
    const normalized = value.replace(/[^0-9.]/g, "");
    const parsed = Number(normalized);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const stats = useMemo(() => {
    const now = Date.now();
    const weekAgo = now - 7 * 24 * 60 * 60 * 1000;
    const recentCount = leads.filter((lead) => {
      const createdAt = lead.created_at ? Date.parse(lead.created_at) : NaN;
      return Number.isFinite(createdAt) && createdAt >= weekAgo;
    }).length;

    const sourceCounts = leads.reduce<Record<string, number>>((acc, lead) => {
      const source = lead.source || "Unknown";
      acc[source] = (acc[source] || 0) + 1;
      return acc;
    }, {});
    const topSource = Object.entries(sourceCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "-";

    const values = leads
      .map((lead) => parseValue(lead.attributes?.estimated_value))
      .filter((value): value is number => value !== null);
    const avgValue = values.length > 0 ? Math.round(values.reduce((sum, v) => sum + v, 0) / values.length) : null;

    return { recentCount, topSource, avgValue };
  }, [leads]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">New Leads</h1>
            <p className="text-slate-600 font-medium mt-1">Recently added leads waiting for follow-up</p>
          </div>
          <button
            onClick={() => setShowAddLeadModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + Add Lead
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">New This Week</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">{loading ? "-" : stats.recentCount}</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Based on created leads</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Lead Value</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">
              {stats.avgValue ? `$${stats.avgValue.toLocaleString()}` : "-"}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">From available lead data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Top Source</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">{loading ? "-" : stats.topSource}</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Based on recent intake</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Conversion Rate</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No conversion data</p>
          </div>
        </div>

        {/* New Leads Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent New Leads</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Lead ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Phone</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Source</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Value</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Date Added</th>
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
                    return (
                      <tr key={lead.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                        <td className="p-4 font-bold text-blue-700">{lead.id}</td>
                        <td className="p-4 font-medium text-slate-900">{name}</td>
                        <td className="p-4 text-slate-700">{contact.email || "-"}</td>
                        <td className="p-4 text-slate-700">{contact.phone || "-"}</td>
                        <td className="p-4">
                          <span className="px-3 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-700">
                            {lead.source || "Unknown"}
                          </span>
                        </td>
                        <td className="p-4 font-bold text-slate-900">{lead.attributes?.estimated_value ?? "-"}</td>
                        <td className="p-4 text-slate-700">{lead.created_at ? String(lead.created_at).slice(0, 10) : "-"}</td>
                        <td className="p-4 space-x-2">
                          <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Contact</button>
                          <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Qualify</button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Add Lead Modal */}
      {showAddLeadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowAddLeadModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-2xl font-bold text-slate-900 mb-4">➕ Add New Lead</h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Full Name *</label>
                  <input
                    type="text"
                    placeholder="John Smith"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Email *</label>
                  <input
                    type="email"
                    placeholder="john@example.com"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Phone</label>
                  <input
                    type="tel"
                    placeholder="+1 (555) 123-4567"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Lead Source</label>
                  <select className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600">
                    <option value="">Select source...</option>
                    <option value="website">🌐 Website</option>
                    <option value="referral">👥 Referral</option>
                    <option value="phone">📞 Phone</option>
                    <option value="email">📧 Email</option>
                    <option value="social">📱 Social Media</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Insurance Interest</label>
                <select className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600">
                  <option value="">Select insurance type...</option>
                  <option value="auto">🚗 Auto Insurance</option>
                  <option value="home">🏠 Home Insurance</option>
                  <option value="life">❤️ Life Insurance</option>
                  <option value="health">⚕️ Health Insurance</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Estimated Value</label>
                <input
                  type="text"
                  placeholder="$5,000"
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Notes</label>
                <textarea
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  rows={3}
                  placeholder="Additional information about the lead..."
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAddLeadModal(false)}
                className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowAddLeadModal(false);
                  // In a real app, this would create the lead
                }}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Add Lead
              </button>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}

