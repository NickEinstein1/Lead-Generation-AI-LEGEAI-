"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { listLeads } from "@/lib/api";

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
    <div className="min-h-screen bg-gradient-to-b from-[#f8fafc] to-[#eef2f7] p-6">
      <div className="mx-auto max-w-6xl">
        <h1 className="text-2xl font-semibold text-neutral-900">Leads</h1>
        <div className="mt-4 rounded border bg-white shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-neutral-50 border-b">
              <tr>
                <th className="text-left p-3">ID</th>
                <th className="text-left p-3">Source</th>
                <th className="text-left p-3">Channel</th>
                <th className="text-left p-3">Product</th>
                <th className="text-left p-3">Created</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={5} className="p-4 text-center text-neutral-500">Loading...</td></tr>
              ) : error ? (
                <tr><td colSpan={5} className="p-4 text-center text-red-600">{error}</td></tr>
              ) : items.length === 0 ? (
                <tr><td colSpan={5} className="p-4 text-center text-neutral-500">No leads</td></tr>
              ) : (
                items.map((l) => (
                  <tr key={l.id} className="border-t hover:bg-neutral-50">
                    <td className="p-3"><Link href={`/dashboard/leads/${l.id}`} className="text-blue-600 underline">{l.id}</Link></td>
                    <td className="p-3">{l.source || "-"}</td>
                    <td className="p-3">{l.channel || "-"}</td>
                    <td className="p-3">{l.product_interest || "-"}</td>
                    <td className="p-3">{l.created_at ? String(l.created_at).slice(0, 19).replace("T", " ") : "-"}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

