"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";

interface Trigger {
  id: number;
  name: string;
  description: string;
  trigger_type: string;
  trigger_config: any;
  is_active: boolean;
  created_at: string;
}

export default function AutomationPage() {
  const router = useRouter();
  const [triggers, setTriggers] = useState<Trigger[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    trigger_type: "time_based",
    trigger_config: {},
    is_active: true,
    created_by: 1,
  });

  useEffect(() => {
    fetchTriggers();
  }, []);

  const fetchTriggers = async () => {
    try {
      const response = await fetch("http://localhost:8000/v1/marketing/triggers");
      const data = await response.json();
      setTriggers(data);
    } catch (error) {
      console.error("Error fetching triggers:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:8000/v1/marketing/triggers", {
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
          trigger_type: "time_based",
          trigger_config: {},
          is_active: true,
          created_by: 1,
        });
        fetchTriggers();
      }
    } catch (error) {
      console.error("Error creating trigger:", error);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this automation trigger?")) return;

    try {
      await fetch(`http://localhost:8000/v1/marketing/triggers/${id}`, {
        method: "DELETE",
      });
      fetchTriggers();
    } catch (error) {
      console.error("Error deleting trigger:", error);
    }
  };

  const getTriggerTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      time_based: "⏰",
      event_based: "🎯",
      behavior_based: "🔄",
      score_based: "📊",
    };
    return icons[type] || "⚡";
  };

  const getTriggerTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      time_based: "bg-blue-100 text-blue-700",
      event_based: "bg-green-100 text-green-700",
      behavior_based: "bg-purple-100 text-purple-700",
      score_based: "bg-amber-100 text-amber-700",
    };
    return colors[type] || "bg-slate-100 text-slate-700";
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-slate-600">Loading automation triggers...</p>
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
              ⚡ Marketing Automation
            </h1>
            <p className="text-slate-600 mt-1">
              Set up automated triggers and workflows for your campaigns
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all flex items-center gap-2"
          >
            <span>➕</span>
            Create Trigger
          </button>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white shadow-md">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-3xl">⏰</span>
              <div>
                <p className="text-sm opacity-90">Time-Based</p>
                <p className="text-2xl font-bold">
                  {triggers.filter((t) => t.trigger_type === "time_based").length}
                </p>
              </div>
            </div>
            <p className="text-xs opacity-75">Scheduled triggers</p>
          </div>

          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-6 text-white shadow-md">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-3xl">🎯</span>
              <div>
                <p className="text-sm opacity-90">Event-Based</p>
                <p className="text-2xl font-bold">
                  {triggers.filter((t) => t.trigger_type === "event_based").length}
                </p>
              </div>
            </div>
            <p className="text-xs opacity-75">Action triggers</p>
          </div>

          <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-6 text-white shadow-md">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-3xl">🔄</span>
              <div>
                <p className="text-sm opacity-90">Behavior-Based</p>
                <p className="text-2xl font-bold">
                  {triggers.filter((t) => t.trigger_type === "behavior_based").length}
                </p>
              </div>
            </div>
            <p className="text-xs opacity-75">User behavior triggers</p>
          </div>

          <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-lg p-6 text-white shadow-md">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-3xl">📊</span>
              <div>
                <p className="text-sm opacity-90">Score-Based</p>
                <p className="text-2xl font-bold">
                  {triggers.filter((t) => t.trigger_type === "score_based").length}
                </p>
              </div>
            </div>
            <p className="text-xs opacity-75">Lead score triggers</p>
          </div>
        </div>

        {/* Triggers List */}
        {triggers.length === 0 ? (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-12 shadow-md text-center">
            <div className="text-6xl mb-4">⚡</div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">No automation triggers yet</h3>
            <p className="text-slate-600 mb-6">
              Create your first automation trigger to start automating your marketing workflows
            </p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Create Trigger
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {triggers.map((trigger) => (
              <div
                key={trigger.id}
                className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <span className="text-3xl">{getTriggerTypeIcon(trigger.trigger_type)}</span>
                    <div>
                      <h3 className="font-bold text-slate-900">{trigger.name}</h3>
                      <span
                        className={`inline-block px-2 py-1 rounded text-xs font-semibold mt-1 ${getTriggerTypeColor(
                          trigger.trigger_type
                        )}`}
                      >
                        {trigger.trigger_type.replace("_", " ").toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <span
                    className={`px-2 py-1 rounded text-xs font-semibold ${
                      trigger.is_active
                        ? "bg-green-100 text-green-700"
                        : "bg-gray-100 text-gray-700"
                    }`}
                  >
                    {trigger.is_active ? "Active" : "Inactive"}
                  </span>
                </div>

                <p className="text-sm text-slate-600 mb-4">{trigger.description}</p>

                <div className="space-y-2 mb-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-600">Created</span>
                    <span className="text-slate-900">
                      {new Date(trigger.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => handleDelete(trigger.id)}
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
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg p-6 max-w-md w-full">
              <h2 className="text-2xl font-bold text-slate-900 mb-4">Create Automation Trigger</h2>
              <form onSubmit={handleCreate} className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Trigger Name *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    required
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                    placeholder="e.g., Welcome Email After Signup"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Description
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    rows={3}
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                    placeholder="Describe this trigger..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Trigger Type *
                  </label>
                  <select
                    value={formData.trigger_type}
                    onChange={(e) => setFormData({ ...formData, trigger_type: e.target.value })}
                    required
                    className="w-full px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  >
                    <option value="time_based">⏰ Time-Based (Scheduled)</option>
                    <option value="event_based">🎯 Event-Based (User Action)</option>
                    <option value="behavior_based">🔄 Behavior-Based (User Behavior)</option>
                    <option value="score_based">📊 Score-Based (Lead Score)</option>
                  </select>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-sm text-slate-700">
                    <strong>Trigger Types:</strong>
                  </p>
                  <ul className="text-xs text-slate-600 mt-2 space-y-1">
                    <li>• <strong>Time-Based:</strong> Trigger at specific times or intervals</li>
                    <li>• <strong>Event-Based:</strong> Trigger when user performs an action</li>
                    <li>• <strong>Behavior-Based:</strong> Trigger based on user behavior patterns</li>
                    <li>• <strong>Score-Based:</strong> Trigger when lead score reaches threshold</li>
                  </ul>
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
                    Create Trigger
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

