"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";

interface Template {
  id: number;
  name: string;
  description: string;
  template_type: string;
  subject_line: string;
  html_content: string;
  text_content: string;
  personalization_tokens: string[];
  is_active: boolean;
  created_at: string;
}

export default function TemplatesPage() {
  const router = useRouter();
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    template_type: "email",
    subject_line: "",
    html_content: "",
    text_content: "",
    personalization_tokens: [],
    is_active: true,
    created_by: 1,
  });

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    try {
      const response = await fetch("http://localhost:8000/v1/marketing/templates");
      const data = await response.json();
      setTemplates(data);
    } catch (error) {
      console.error("Error fetching templates:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:8000/v1/marketing/templates", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        setShowCreateModal(false);
        setFormData({
          name: "",
          description: "",
          template_type: "email",
          subject_line: "",
          html_content: "",
          text_content: "",
          personalization_tokens: [],
          is_active: true,
          created_by: 1,
        });
        fetchTemplates();
      }
    } catch (error) {
      console.error("Error creating template:", error);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this template?")) return;

    try {
      await fetch(`http://localhost:8000/v1/marketing/templates/${id}`, {
        method: "DELETE",
      });
      fetchTemplates();
    } catch (error) {
      console.error("Error deleting template:", error);
    }
  };

  const getTemplateTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      email: "📧",
      sms: "💬",
      multi_channel: "🌐",
      drip: "💧",
    };
    return icons[type] || "📝";
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-slate-600">Loading templates...</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-2">
              📝 Marketing Templates
            </h1>
            <p className="text-slate-600 mt-1">
              Create and manage email and SMS templates for your campaigns
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all flex items-center gap-2"
          >
            <span>➕</span>
            Create Template
          </button>
        </div>

        {/* Templates Grid */}
        {templates.length === 0 ? (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-12 shadow-md text-center">
            <div className="text-6xl mb-4">📝</div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">No templates yet</h3>
            <p className="text-slate-600 mb-6">
              Create your first template to use in your marketing campaigns
            </p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Create Template
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {templates.map((template) => (
              <div
                key={template.id}
                className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">{getTemplateTypeIcon(template.template_type)}</span>
                    <div>
                      <h3 className="font-bold text-slate-900">{template.name}</h3>
                      <p className="text-xs text-slate-500 capitalize">{template.template_type}</p>
                    </div>
                  </div>
                  <span
                    className={`px-2 py-1 rounded text-xs font-semibold ${
                      template.is_active
                        ? "bg-green-100 text-green-700"
                        : "bg-gray-100 text-gray-700"
                    }`}
                  >
                    {template.is_active ? "Active" : "Inactive"}
                  </span>
                </div>

                <p className="text-sm text-slate-600 mb-3">{template.description}</p>

                {template.subject_line && (
                  <div className="bg-blue-50 border border-blue-200 rounded p-2 mb-3">
                    <p className="text-xs text-slate-600 mb-1">Subject Line:</p>
                    <p className="text-sm font-semibold text-slate-900">{template.subject_line}</p>
                  </div>
                )}

                <div className="space-y-2 mb-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-600">Created</span>
                    <span className="text-slate-900">
                      {new Date(template.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  {template.personalization_tokens && template.personalization_tokens.length > 0 && (
                    <div>
                      <p className="text-xs text-slate-600 mb-1">Tokens:</p>
                      <div className="flex flex-wrap gap-1">
                        {template.personalization_tokens.slice(0, 3).map((token, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-mono"
                          >
                            {token}
                          </span>
                        ))}
                        {template.personalization_tokens.length > 3 && (
                          <span className="px-2 py-1 bg-slate-100 text-slate-600 rounded text-xs">
                            +{template.personalization_tokens.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => handleDelete(template.id)}
                    className="flex-1 bg-red-50 text-red-600 px-4 py-2 rounded-lg font-semibold hover:bg-red-100 transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Create Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full my-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-4">Create Marketing Template</h2>
              <form onSubmit={handleCreate} className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Template Name *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    required
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                    placeholder="e.g., Spring Auto Insurance Promotion"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Description
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    rows={2}
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                    placeholder="Describe this template..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Template Type *
                  </label>
                  <select
                    value={formData.template_type}
                    onChange={(e) => setFormData({ ...formData, template_type: e.target.value })}
                    required
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  >
                    <option value="email">📧 Email Template</option>
                    <option value="sms">💬 SMS Template</option>
                    <option value="multi_channel">🌐 Multi-Channel Template</option>
                    <option value="drip">💧 Drip Template</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Subject Line
                  </label>
                  <input
                    type="text"
                    value={formData.subject_line}
                    onChange={(e) => setFormData({ ...formData, subject_line: e.target.value })}
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                    placeholder="e.g., Save 20% on Auto Insurance!"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    HTML Content
                  </label>
                  <textarea
                    value={formData.html_content}
                    onChange={(e) => setFormData({ ...formData, html_content: e.target.value })}
                    rows={4}
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm"
                    placeholder="<html>...</html>"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Text Content
                  </label>
                  <textarea
                    value={formData.text_content}
                    onChange={(e) => setFormData({ ...formData, text_content: e.target.value })}
                    rows={3}
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                    placeholder="Plain text version..."
                  />
                </div>

                <div className="flex gap-2 pt-4">
                  <button
                    type="button"
                    onClick={() => setShowCreateModal(false)}
                    className="flex-1 bg-slate-100 text-slate-700 px-4 py-2 rounded-lg font-semibold hover:bg-slate-200 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                  >
                    Create Template
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

