"use client";
import { useState } from "react";
import { createLead, scoreLead, routeLead } from "@/lib/api";

export default function Home() {
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
      geo: {
        country: "US",
        state: getStr(formData.get("state")),
      },
      attributes: {},
      consent: { timestamp: new Date().toISOString(), method: "web_form" },
    };
    try {
      const res = await createLead(payload);
      setStatus(res.status);
      setLeadId(res.lead_id);
      form.reset();
    } catch (err: unknown) {
      if (err instanceof Error) {
        setStatus(err.message);
      } else {
        setStatus("Failed");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">Insurance Lead Intake</h1>
      <form onSubmit={onSubmit} className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <input name="first_name" placeholder="First name" className="border px-3 py-2 rounded" />
          <input name="last_name" placeholder="Last name" className="border px-3 py-2 rounded" />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <input name="email" type="email" placeholder="Email" className="border px-3 py-2 rounded" />
          <input name="phone" placeholder="Phone" className="border px-3 py-2 rounded" />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <select name="product_interest" className="border px-3 py-2 rounded">
            <option value="">Select product</option>
            <option value="auto">Auto</option>
            <option value="home">Home</option>
            <option value="life">Life</option>
            <option value="health">Health</option>
          </select>
          <input name="state" placeholder="State (e.g., CA)" className="border px-3 py-2 rounded" />
        </div>
        <button type="submit" disabled={loading} className="bg-black text-white px-4 py-2 rounded">
          {loading ? "Submitting..." : "Submit Lead"}
        </button>
      </form>
      {status && (
        <div className="mt-4 text-sm space-y-2">
          <div>Status: {status}</div>
          {leadId && (
            <div className="space-y-2">
              <div>
                Lead ID: <code className="px-1 py-0.5 bg-gray-100 rounded">{leadId}</code>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={async () => {
                    setLoading(true);
                    try {
                      const res = await scoreLead(leadId, {});
                      setStatus(`Scored: ${res.score} (band=${res.band})`);
                    } catch (e: unknown) {
                      if (e instanceof Error) setStatus(e.message);
                      else setStatus("Score failed");
                    } finally {
                      setLoading(false);
                    }
                  }}
                  className="border px-3 py-1 rounded"
                >
                  Score Lead
                </button>
                <button
                  onClick={async () => {
                    setLoading(true);
                    try {
                      const res = await routeLead(leadId);
                      setStatus(`Routed to agent ${res.agent_id}`);
                    } catch (e: unknown) {
                      if (e instanceof Error) setStatus(e.message);
                      else setStatus("Route failed");
                    } finally {
                      setLoading(false);
                    }
                  }}
                  className="border px-3 py-1 rounded"
                >
                  Route Lead
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
