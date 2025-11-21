"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";

interface Segment {
  id: number;
  name: string;
  estimated_size: number;
}

interface Template {
  id: number;
  name: string;
  template_type: string;
  subject_line: string;
}

export default function CreateCampaignPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState({
    name: "",
    description: "",
    campaign_type: "email",
    segment_id: "",
    template_id: "",
    subject_line: "",
    scheduled_at: "",
    is_ab_test: false,
    is_automated: false,
  });

  useEffect(() => {
    fetchSegments();
    fetchTemplates();
  }, []);

  const fetchSegments = async () => {
    try {
      const response = await fetch("http://localhost:8000/v1/marketing/segments");
      const data = await response.json();
      setSegments(data);
    } catch (error) {
      console.error("Error fetching segments:", error);
    }
  };

  const fetchTemplates = async () => {
    try {
      const response = await fetch("http://localhost:8000/v1/marketing/templates");
      const data = await response.json();
      setTemplates(data);
    } catch (error) {
      console.error("Error fetching templates:", error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const payload = {
        ...formData,
        segment_id: formData.segment_id ? parseInt(formData.segment_id) : null,
        template_id: formData.template_id ? parseInt(formData.template_id) : null,
        scheduled_at: formData.scheduled_at || null,
        status: "draft",
        created_by: 1, // TODO: Get from auth context
      };

      const response = await fetch("http://localhost:8000/v1/marketing/campaigns", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        const campaign = await response.json();
        router.push(`/dashboard/marketing/campaigns/${campaign.id}`);
      } else {
        alert("Error creating campaign");
      }
    } catch (error) {
      console.error("Error creating campaign:", error);
      alert("Error creating campaign");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? (e.target as HTMLInputElement).checked : value,
    }));
  };

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-2">
              ➕ Create Campaign
            </h1>
            <p className="text-slate-600 mt-1">
              Set up a new marketing campaign in {step} of 3 steps
            </p>
          </div>
          <button
            onClick={() => router.push("/dashboard/marketing/campaigns")}
            className="text-slate-600 hover:text-slate-900 font-semibold"
          >
            ← Back to Campaigns
          </button>
        </div>

        {/* Progress Steps */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <div className="flex items-center justify-between">
            <div className={`flex items-center gap-3 ${step >= 1 ? "text-blue-600" : "text-slate-400"}`}>
              <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                step >= 1 ? "bg-blue-600 text-white" : "bg-slate-200"
              }`}>
                1
              </div>
              <span className="font-semibold">Campaign Details</span>
            </div>
            <div className="flex-1 h-1 mx-4 bg-slate-200">
              <div className={`h-full ${step >= 2 ? "bg-blue-600" : "bg-slate-200"}`} style={{ width: step >= 2 ? "100%" : "0%" }}></div>
            </div>
            <div className={`flex items-center gap-3 ${step >= 2 ? "text-blue-600" : "text-slate-400"}`}>
              <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                step >= 2 ? "bg-blue-600 text-white" : "bg-slate-200"
              }`}>
                2
              </div>
              <span className="font-semibold">Audience & Template</span>
            </div>
            <div className="flex-1 h-1 mx-4 bg-slate-200">
              <div className={`h-full ${step >= 3 ? "bg-blue-600" : "bg-slate-200"}`} style={{ width: step >= 3 ? "100%" : "0%" }}></div>
            </div>
            <div className={`flex items-center gap-3 ${step >= 3 ? "text-blue-600" : "text-slate-400"}`}>
              <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                step >= 3 ? "bg-blue-600 text-white" : "bg-slate-200"
              }`}>
                3
              </div>
              <span className="font-semibold">Schedule & Launch</span>
            </div>
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit}>
          {/* Step 1: Campaign Details */}
          {step === 1 && (
            <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md space-y-6">
              <h2 className="text-xl font-bold text-slate-900">Campaign Details</h2>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Campaign Name *
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  placeholder="e.g., Spring Insurance Promotion"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Description
                </label>
                <textarea
                  name="description"
                  value={formData.description}
                  onChange={handleChange}
                  rows={3}
                  className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  placeholder="Describe your campaign..."
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Campaign Type *
                </label>
                <select
                  name="campaign_type"
                  value={formData.campaign_type}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                >
                  <option value="email">📧 Email Campaign</option>
                  <option value="sms">💬 SMS Campaign</option>
                  <option value="multi_channel">🌐 Multi-Channel Campaign</option>
                  <option value="drip">💧 Drip Campaign</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Subject Line
                </label>
                <input
                  type="text"
                  name="subject_line"
                  value={formData.subject_line}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  placeholder="e.g., Save 20% on Auto Insurance This Spring!"
                />
              </div>

              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    name="is_ab_test"
                    checked={formData.is_ab_test}
                    onChange={handleChange}
                    className="w-5 h-5 text-blue-600"
                  />
                  <span className="text-sm font-semibold text-slate-700">Enable A/B Testing</span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    name="is_automated"
                    checked={formData.is_automated}
                    onChange={handleChange}
                    className="w-5 h-5 text-blue-600"
                  />
                  <span className="text-sm font-semibold text-slate-700">Automated Campaign</span>
                </label>
              </div>

              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => setStep(2)}
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                >
                  Next: Audience & Template →
                </button>
              </div>
            </div>
          )}

          {/* Step 2: Audience & Template */}
          {step === 2 && (
            <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md space-y-6">
              <h2 className="text-xl font-bold text-slate-900">Audience & Template</h2>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Target Audience Segment
                </label>
                <select
                  name="segment_id"
                  value={formData.segment_id}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                >
                  <option value="">Select a segment...</option>
                  {segments.map((segment) => (
                    <option key={segment.id} value={segment.id}>
                      {segment.name} ({segment.estimated_size?.toLocaleString() || 0} contacts)
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-500 mt-1">
                  Don't have a segment? <button type="button" onClick={() => router.push("/dashboard/marketing/segments")} className="text-blue-600 hover:underline">Create one</button>
                </p>
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Email/SMS Template
                </label>
                <select
                  name="template_id"
                  value={formData.template_id}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                >
                  <option value="">Select a template...</option>
                  {templates
                    .filter((t) => t.template_type === formData.campaign_type)
                    .map((template) => (
                      <option key={template.id} value={template.id}>
                        {template.name} - {template.subject_line}
                      </option>
                    ))}
                </select>
                <p className="text-xs text-slate-500 mt-1">
                  Need a template? <button type="button" onClick={() => router.push("/dashboard/marketing/templates")} className="text-blue-600 hover:underline">Create one</button>
                </p>
              </div>

              <div className="flex justify-between">
                <button
                  type="button"
                  onClick={() => setStep(1)}
                  className="text-slate-600 hover:text-slate-900 font-semibold"
                >
                  ← Back
                </button>
                <button
                  type="button"
                  onClick={() => setStep(3)}
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                >
                  Next: Schedule & Launch →
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Schedule & Launch */}
          {step === 3 && (
            <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md space-y-6">
              <h2 className="text-xl font-bold text-slate-900">Schedule & Launch</h2>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Schedule Campaign (Optional)
                </label>
                <input
                  type="datetime-local"
                  name="scheduled_at"
                  value={formData.scheduled_at}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                />
                <p className="text-xs text-slate-500 mt-1">
                  Leave empty to save as draft
                </p>
              </div>

              <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
                <h3 className="font-bold text-slate-900 mb-2">Campaign Summary</h3>
                <div className="space-y-1 text-sm">
                  <p><span className="font-semibold">Name:</span> {formData.name}</p>
                  <p><span className="font-semibold">Type:</span> {formData.campaign_type.replace("_", " ").toUpperCase()}</p>
                  <p><span className="font-semibold">Subject:</span> {formData.subject_line || "Not set"}</p>
                  <p><span className="font-semibold">A/B Testing:</span> {formData.is_ab_test ? "Yes" : "No"}</p>
                  <p><span className="font-semibold">Automated:</span> {formData.is_automated ? "Yes" : "No"}</p>
                </div>
              </div>

              <div className="flex justify-between">
                <button
                  type="button"
                  onClick={() => setStep(2)}
                  className="text-slate-600 hover:text-slate-900 font-semibold"
                >
                  ← Back
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-8 py-3 rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50"
                >
                  {loading ? "Creating..." : "Create Campaign"}
                </button>
              </div>
            </div>
          )}
        </form>
      </div>
    </DashboardLayout>
  );
}

