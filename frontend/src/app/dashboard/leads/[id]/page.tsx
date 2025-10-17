"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getLead, scoreLead } from "@/lib/api";

export default function LeadDetailPage() {
  const params = useParams();
  const id = String(params?.id || "");
  const [lead, setLead] = useState<any>(null);
  const [error, setError] = useState<string>("");
  const [scoring, setScoring] = useState(false);

  useEffect(() => {
    if (!id) return;
    (async () => {
      try {
        const data = await getLead(id);
        setLead(data);
      } catch (e: any) {
        setError(e?.message || "Failed to load lead");
      }
    })();
  }, [id]);

  async function onScore() {
    try {
      setScoring(true);
      const res = await scoreLead(id, {});
      alert(`Scored: ${res.score} (${res.band})`);
    } catch (e: any) {
      alert(e?.message || "Score failed");
    } finally {
      setScoring(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#f8fafc] to-[#eef2f7] p-6">
      <div className="mx-auto max-w-4xl space-y-4">
        <h1 className="text-2xl font-semibold text-neutral-900">Lead {id}</h1>
        {error && <div className="text-red-600">{error}</div>}
        {!lead ? (
          <div className="text-neutral-600">Loading...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded border bg-white p-4 shadow-sm">
              <div className="font-medium mb-2">Details</div>
              <div className="text-sm text-neutral-700 space-y-1">
                <div><span className="text-neutral-500">Source:</span> {lead.source || "-"}</div>
                <div><span className="text-neutral-500">Channel:</span> {lead.channel || "-"}</div>
                <div><span className="text-neutral-500">Product:</span> {lead.product_interest || "-"}</div>
                <div><span className="text-neutral-500">Email:</span> {lead.contact?.email || "-"}</div>
              </div>
            </div>
            <div className="rounded border bg-white p-4 shadow-sm">
              <div className="font-medium mb-2">Actions</div>
              <button disabled={scoring} onClick={onScore} className="bg-primary text-white rounded px-3 py-2">
                {scoring ? "Scoring..." : "Score lead"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

