"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { listLeads } from "@/lib/api";
import DashboardLayout from "@/components/DashboardLayout";

export default function LeadsListPage() {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const res = await listLeads(50, 0);
        setItems(res.items || []);
      } catch (e: any) {
        setError(e?.message || "Failed to load leads");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <DashboardLayout>
      <div className="mx-auto max-w-6xl">
        <h1 className="text-2xl font-bold text-slate-900">Leads</h1>
        <div className="mt-4 rounded border-2 border-blue-200 bg-white shadow-md overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-blue-50 border-b-2 border-blue-200">
              <tr>
                <th className="text-left p-3 text-slate-900 font-bold">ID</th>
                <th className="text-left p-3 text-slate-900 font-bold">Source</th>
                <th className="text-left p-3 text-slate-900 font-bold">Channel</th>
                <th className="text-left p-3 text-slate-900 font-bold">Product</th>
                <th className="text-left p-3 text-slate-900 font-bold">Created</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={5} className="p-4 text-center text-slate-600 font-medium">Loading...</td></tr>
              ) : error ? (
                <tr><td colSpan={5} className="p-4 text-center text-red-600 font-medium">{error}</td></tr>
              ) : items.length === 0 ? (
                <tr><td colSpan={5} className="p-4 text-center text-slate-600 font-medium">No leads</td></tr>
              ) : (
                items.map((l) => (
                  <tr key={l.id} className="border-t border-blue-100 hover:bg-blue-50">
                    <td className="p-3"><Link href={`/dashboard/leads/${l.id}`} className="text-blue-700 hover:text-blue-800 underline font-medium">{l.id}</Link></td>
                    <td className="p-3 text-slate-700 font-medium">{l.source || "-"}</td>
                    <td className="p-3 text-slate-700 font-medium">{l.channel || "-"}</td>
                    <td className="p-3 text-slate-700 font-medium">{l.product_interest || "-"}</td>
                    <td className="p-3 text-slate-700 font-medium">{l.created_at ? String(l.created_at).slice(0, 19).replace("T", " ") : "-"}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </DashboardLayout>
  );
}

