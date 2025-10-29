"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { getLead, scoreLead, listDocumentsForLead, createDocumentForLead, simulateSignDocument } from "@/lib/api";
import LeadInfo from "@/components/LeadInfo";
import LeadActions from "@/components/LeadActions";
import DocumentsManager from "@/components/DocumentsManager";
import LeadManagement from "@/components/LeadManagement";
import DashboardLayout from "@/components/DashboardLayout";

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
    <DashboardLayout>
      <div className="mx-auto max-w-6xl space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-slate-900">Lead Details</h1>
          <button
            onClick={() => window.history.back()}
            className="text-slate-600 hover:text-slate-900 font-medium"
          >
            ← Back
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-2 border-red-200 text-red-700 p-4 rounded-lg font-medium">
            {error}
          </div>
        )}

        {/* Loading State */}
        {!lead ? (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-8 text-center">
            <p className="text-slate-600 font-medium">Loading lead details...</p>
          </div>
        ) : (
          <>
            {/* Lead Information */}
            <section>
              <LeadInfo lead={lead} />
            </section>

            {/* Main Grid: Actions and Documents */}
            <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <LeadActions
                  leadId={id}
                  onScore={onScore}
                  onCreateDoc={onCreateDoc}
                  scoring={scoring}
                  creating={creating}
                />
              </div>
              <div className="lg:col-span-2">
                <DocumentsManager
                  documents={docs}
                  onCreateDoc={onCreateDoc}
                  onOpenSign={onOpenSign}
                  onSimulateSign={onSimulateSign}
                  creating={creating}
                />
              </div>
            </section>

            {/* Activity Timeline */}
            <section>
              <LeadManagement />
            </section>
          </>
        )}
      </div>
    </DashboardLayout>
  );
}

