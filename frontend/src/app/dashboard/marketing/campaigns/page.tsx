"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";

interface Campaign {
  id: number;
  name: string;
  description: string;
  campaign_type: string;
  status: string;
  segment_id: number | null;
  template_id: number | null;
  subject_line: string;
  scheduled_at: string | null;
  is_ab_test: boolean;
  is_automated: boolean;
  created_at: string;
  updated_at: string;
}

export default function CampaignsPage() {
  const router = useRouter();
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    fetchCampaigns();
  }, [filter]);

  const fetchCampaigns = async () => {
    try {
      let url = "http://localhost:8000/v1/marketing/campaigns?limit=100";
      if (filter !== "all") {
        url += `&status=${filter}`;
      }
      const response = await fetch(url);
      const data = await response.json();
      setCampaigns(data);
    } catch (error) {
      console.error("Error fetching campaigns:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this campaign?")) return;
    
    try {
      await fetch(`http://localhost:8000/v1/marketing/campaigns/${id}`, {
        method: "DELETE",
      });
      fetchCampaigns();
    } catch (error) {
      console.error("Error deleting campaign:", error);
    }
  };

  const handleLaunch = async (id: number) => {
    try {
      await fetch(`http://localhost:8000/v1/marketing/campaigns/${id}/launch`, {
        method: "POST",
      });
      fetchCampaigns();
    } catch (error) {
      console.error("Error launching campaign:", error);
    }
  };

  const handlePause = async (id: number) => {
    try {
      await fetch(`http://localhost:8000/v1/marketing/campaigns/${id}/pause`, {
        method: "POST",
      });
      fetchCampaigns();
    } catch (error) {
      console.error("Error pausing campaign:", error);
    }
  };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      draft: "bg-gray-100 text-gray-700",
      scheduled: "bg-blue-100 text-blue-700",
      active: "bg-green-100 text-green-700",
      paused: "bg-yellow-100 text-yellow-700",
      completed: "bg-purple-100 text-purple-700",
      archived: "bg-slate-100 text-slate-700",
    };
    return colors[status] || "bg-gray-100 text-gray-700";
  };

  const getCampaignTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      email: "📧",
      sms: "💬",
      multi_channel: "🌐",
      drip: "💧",
    };
    return icons[type] || "📢";
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-slate-600">Loading campaigns...</p>
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
              📧 Marketing Campaigns
            </h1>
            <p className="text-slate-600 mt-1">
              Create, manage, and track all your marketing campaigns
            </p>
          </div>
          <button
            onClick={() => router.push("/dashboard/marketing/campaigns/create")}
            className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all flex items-center gap-2"
          >
            <span>➕</span>
            Create Campaign
          </button>
        </div>

        {/* Filters */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setFilter("all")}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                filter === "all"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              All Campaigns
            </button>
            <button
              onClick={() => setFilter("draft")}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                filter === "draft"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              Draft
            </button>
            <button
              onClick={() => setFilter("scheduled")}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                filter === "scheduled"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              Scheduled
            </button>
            <button
              onClick={() => setFilter("active")}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                filter === "active"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              Active
            </button>
            <button
              onClick={() => setFilter("paused")}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                filter === "paused"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              Paused
            </button>
            <button
              onClick={() => setFilter("completed")}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                filter === "completed"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              Completed
            </button>
          </div>
        </div>

        {/* Campaigns List */}
        {campaigns.length === 0 ? (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-12 shadow-md text-center">
            <div className="text-6xl mb-4">📧</div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">No campaigns found</h3>
            <p className="text-slate-600 mb-6">
              {filter === "all"
                ? "Create your first marketing campaign to get started"
                : `No ${filter} campaigns found. Try a different filter.`}
            </p>
            <button
              onClick={() => router.push("/dashboard/marketing/campaigns/create")}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Create Campaign
            </button>
          </div>
        ) : (
          <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-blue-50 border-b-2 border-blue-200">
                  <tr>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Campaign</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Type</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Status</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Features</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Scheduled</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Created</th>
                    <th className="text-right py-3 px-4 font-semibold text-slate-700">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {campaigns.map((campaign) => (
                    <tr key={campaign.id} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="py-4 px-4">
                        <div>
                          <p className="font-semibold text-slate-900">{campaign.name}</p>
                          <p className="text-sm text-slate-600">{campaign.description}</p>
                          {campaign.subject_line && (
                            <p className="text-xs text-slate-500 mt-1">
                              Subject: {campaign.subject_line}
                            </p>
                          )}
                        </div>
                      </td>
                      <td className="py-4 px-4">
                        <span className="flex items-center gap-2">
                          {getCampaignTypeIcon(campaign.campaign_type)}
                          <span className="capitalize">{campaign.campaign_type.replace("_", " ")}</span>
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(campaign.status)}`}>
                          {campaign.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <div className="flex gap-1">
                          {campaign.is_ab_test && (
                            <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-semibold">
                              A/B
                            </span>
                          )}
                          {campaign.is_automated && (
                            <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-semibold">
                              Auto
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-4 px-4 text-slate-600 text-sm">
                        {campaign.scheduled_at
                          ? new Date(campaign.scheduled_at).toLocaleString()
                          : "-"}
                      </td>
                      <td className="py-4 px-4 text-slate-600 text-sm">
                        {new Date(campaign.created_at).toLocaleDateString()}
                      </td>
                      <td className="py-4 px-4">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => router.push(`/dashboard/marketing/campaigns/${campaign.id}`)}
                            className="text-blue-600 hover:text-blue-700 font-semibold text-sm"
                          >
                            View
                          </button>
                          {campaign.status === "draft" && (
                            <button
                              onClick={() => handleLaunch(campaign.id)}
                              className="text-green-600 hover:text-green-700 font-semibold text-sm"
                            >
                              Launch
                            </button>
                          )}
                          {campaign.status === "active" && (
                            <button
                              onClick={() => handlePause(campaign.id)}
                              className="text-yellow-600 hover:text-yellow-700 font-semibold text-sm"
                            >
                              Pause
                            </button>
                          )}
                          {campaign.status === "paused" && (
                            <button
                              onClick={() => handleLaunch(campaign.id)}
                              className="text-green-600 hover:text-green-700 font-semibold text-sm"
                            >
                              Resume
                            </button>
                          )}
                          <button
                            onClick={() => handleDelete(campaign.id)}
                            className="text-red-600 hover:text-red-700 font-semibold text-sm"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

