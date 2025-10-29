"use client";
import { useState } from "react";
import { createLead, scoreLead, routeLead } from "@/lib/api";

export default function NewLeadPage() {
  const [status, setStatus] = useState<string>("");
  const [leadId, setLeadId] = useState<string>("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("");
    setLeadId("");
    setLoading(true);
    const form = e.currentTarget;
    const formData = new FormData(form);
    const getStr = (v: FormDataEntryValue | null) => (typeof v === "string" && v.length ? v : null);
    const payload = {
      idempotency_key: crypto.randomUUID(),
      source: "web",
      channel: "web",
      product_interest: getStr(formData.get("product_interest")),
      contact: {
        first_name: getStr(formData.get("first_name")),
        last_name: getStr(formData.get("last_name")),
        email: getStr(formData.get("email")),
        phone: getStr(formData.get("phone")),
      },
      geo: { country: "US", state: getStr(formData.get("state")) },
      attributes: {},
      consent: { timestamp: new Date().toISOString(), method: "web_form" },
    };
    try {
      const res = await createLead(payload);
      setStatus(res.status);
      setLeadId(res.lead_id);
      form.reset();
    } catch (err: unknown) {
      setStatus(err instanceof Error ? err.message : "Failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-slate-900 mb-4">New Lead</h1>
      <form onSubmit={onSubmit} className="space-y-4 bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
        <div className="grid grid-cols-2 gap-4">
          <input name="first_name" placeholder="First name" className="border-2 border-blue-200 px-3 py-2 rounded text-slate-900 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium" />
          <input name="last_name" placeholder="Last name" className="border-2 border-blue-200 px-3 py-2 rounded text-slate-900 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium" />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <input name="email" type="email" placeholder="Email" className="border-2 border-blue-200 px-3 py-2 rounded text-slate-900 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium" />
          <input name="phone" placeholder="Phone" className="border-2 border-blue-200 px-3 py-2 rounded text-slate-900 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium" />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <select name="product_interest" className="border-2 border-blue-200 px-3 py-2 rounded text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium">
            <option value="">Select product</option>
            <option value="auto">Auto</option>
            <option value="home">Home</option>
            <option value="life">Life</option>
            <option value="health">Health</option>
          </select>
          <input name="state" placeholder="State (e.g., CA)" className="border-2 border-blue-200 px-3 py-2 rounded text-slate-900 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium" />
        </div>
        <button type="submit" disabled={loading} className="bg-blue-700 text-white px-4 py-2 rounded hover:bg-blue-800 font-medium disabled:opacity-50 w-full">
          {loading ? "Submitting..." : "Submit Lead"}
        </button>
      </form>
      {status && (
        <div className="mt-4 text-sm space-y-2 bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <div className="text-slate-900 font-bold">Status: <span className="text-blue-700">{status}</span></div>
          {leadId && (
            <div className="space-y-2">
              <div className="text-slate-900 font-medium">
                Lead ID: <code className="px-2 py-1 bg-blue-50 border border-blue-200 rounded text-blue-700 font-mono">{leadId}</code>
              </div>
              <div className="flex gap-2">
                <button onClick={async () => {
                  setLoading(true);
                  try {
                    const res = await scoreLead(leadId, {});
                    setStatus(`Scored: ${res.score} (band=${res.band})`);
                  } catch (e: unknown) {
                    setStatus(e instanceof Error ? e.message : "Score failed");
                  } finally { setLoading(false); }
                }} className="border-2 border-blue-700 text-blue-700 px-3 py-1 rounded hover:bg-blue-50 font-medium">Score Lead</button>
                <button onClick={async () => {
                  setLoading(true);
                  try {
                    const res = await routeLead(leadId);
                    setStatus(`Routed to agent ${res.agent_id}`);
                  } catch (e: unknown) {
                    setStatus(e instanceof Error ? e.message : "Route failed");
                  } finally { setLoading(false); }
                }} className="border-2 border-blue-700 text-blue-700 px-3 py-1 rounded hover:bg-blue-50 font-medium">Route Lead</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

