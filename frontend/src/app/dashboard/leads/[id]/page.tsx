"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getLead, scoreLead, listDocumentsForLead, createDocumentForLead, simulateSignDocument } from "@/lib/api";

export default function LeadDetailPage() {
  const params = useParams();
  const id = String(params?.id || "");
  const [lead, setLead] = useState<any>(null);
  const [error, setError] = useState<string>("");
  const [scoring, setScoring] = useState(false);

  const [docs, setDocs] = useState<any[]>([]);
  const [creating, setCreating] = useState(false);

  async function fetchDocs() {
    if (!id) return;
    try {
      const d = await listDocumentsForLead(id);
      const items = Array.isArray(d) ? d : (d?.items || []);
      setDocs(items);
    } catch (_) {}
  }

  useEffect(() => {
    if (!id) return;
    (async () => {
      try {
        const data = await getLead(id);
        setLead(data);
      } catch (e: any) {
        setError(e?.message || "Failed to load lead");
      }
      await fetchDocs();
    })();
  }, [id]);

  async function onCreateDoc() {
    try {
      setCreating(true);
      await createDocumentForLead(id, "Insurance Agreement", { provider: 'docuseal' });
      await fetchDocs();
    } catch (e: any) {
      alert(e?.message || "Create document failed");
    } finally {
      setCreating(false);
    }
  }

  function onOpenSign(url?: string) {
    if (!url) return alert("No signing URL available yet");
    window.open(url, "_blank", "noopener,noreferrer");
  }

  async function onSimulateSign(docId: number) {
    try {
      await simulateSignDocument(docId);
      await fetchDocs();
    } catch (e: any) {
      alert(e?.message || "Sign failed");
    }
  }

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

            <div className="rounded border bg-white p-4 shadow-sm md:col-span-2">
              <div className="flex items-center justify-between mb-2">
                <div className="font-medium">Documents</div>
                <button onClick={onCreateDoc} disabled={creating} className="text-sm bg-primary text-white px-3 py-1.5 rounded">
                  {creating ? "Creating..." : "+ Create"}
                </button>
              </div>
              {docs.length === 0 ? (
                <div className="text-sm text-neutral-600">No documents yet.</div>
              ) : (
                <ul className="text-sm space-y-2">
                  {docs.map((d: any) => (
                    <li key={d.id} className="flex items-center justify-between rounded border p-2">
                      <div>
                        <div className="font-medium">{d.title}</div>
                        <div className="text-xs text-neutral-600 flex items-center gap-2">
                          <span className={
                            d.status === 'signed' ? 'text-green-700' : d.status === 'declined' ? 'text-red-700' : 'text-amber-700'
                          }>
                            {d.status}
                          </span>
                          {d.signed_at && <span className="text-neutral-500">• {new Date(d.signed_at).toLocaleString()}</span>}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {d.status !== 'signed' && d.signing_url && (
                          <button onClick={() => onOpenSign(d.signing_url)} className="text-xs border px-2 py-1 rounded">Sign Document</button>
                        )}
                        {d.provider === 'internal' && d.status !== 'signed' && (
                          <button onClick={() => onSimulateSign(d.id)} className="text-xs border px-2 py-1 rounded">Simulate sign</button>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

