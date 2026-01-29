"use client";
import { useEffect, useMemo, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { listLeads } from "@/lib/api";

export default function ContactedLeadsPage() {
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
    return { total };
  }, [leads]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Contacted Leads</h1>
            <p className="text-slate-600 font-medium mt-1">Leads that have been contacted and are in follow-up</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Log Contact
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Contacted</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">{loading ? "-" : stats.total}</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Based on available leads</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Interested</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No status data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Response Time</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No response data</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Follow-ups Due</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No follow-up data</p>
          </div>
        </div>

        {/* Contacted Leads Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Contact History & Follow-ups</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Lead ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Last Contact</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Next Follow-up</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={7} className="p-4 text-center text-slate-600 font-medium">Loading leads...</td>
                  </tr>
                ) : error ? (
                  <tr>
                    <td colSpan={7} className="p-4 text-center text-red-600 font-medium">{error}</td>
                  </tr>
                ) : leads.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="p-4 text-center text-slate-600 font-medium">No leads available.</td>
                  </tr>
                ) : (
                  leads.map((lead) => {
                    const contact = lead.contact || lead.contact_info || {};
                    const name = [contact.first_name, contact.last_name].filter(Boolean).join(" ") || "-";
                    const status = lead.status || "Unknown";
                    const statusClass = status === "Interested"
                      ? "bg-emerald-100 text-emerald-700"
                      : status === "Considering"
                        ? "bg-amber-100 text-amber-700"
                        : "bg-red-100 text-red-700";
                    return (
                      <tr key={lead.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                        <td className="p-4 font-bold text-blue-700">{lead.id}</td>
                        <td className="p-4 font-medium text-slate-900">{name}</td>
                        <td className="p-4 text-slate-700">{contact.email || "-"}</td>
                        <td className="p-4 text-slate-700">{lead.last_contacted_at ? String(lead.last_contacted_at).slice(0, 10) : "-"}</td>
                        <td className="p-4 text-slate-700">{lead.next_follow_up_at ? String(lead.next_follow_up_at).slice(0, 10) : "-"}</td>
                        <td className="p-4">
                          <span className={`px-3 py-1 rounded-full text-xs font-bold ${statusClass}`}>
                            {status}
                          </span>
                        </td>
                        <td className="p-4 space-x-2">
                          <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Follow-up</button>
                          <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">History</button>
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
    </DashboardLayout>
  );
}

