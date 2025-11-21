"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
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

interface Analytics {
  total_sent: number;
  total_delivered: number;
  total_bounced: number;
  total_failed: number;
  total_opened: number;
  unique_opened: number;
  total_clicked: number;
  unique_clicked: number;
  total_conversions: number;
  total_revenue: number;
  delivery_rate: number;
  open_rate: number;
  click_rate: number;
  conversion_rate: number;
  roi: number;
}

export default function CampaignDetailPage() {
  const router = useRouter();
  const params = useParams();
  const id = params?.id as string;

  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id) {
      fetchCampaign();
      fetchAnalytics();
    }
  }, [id]);

  const fetchCampaign = async () => {
    try {
      const response = await fetch(`http://localhost:8000/v1/marketing/campaigns/${id}`);
      const data = await response.json();
      setCampaign(data);
    } catch (error) {
      console.error("Error fetching campaign:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const response = await fetch(`http://localhost:8000/v1/marketing/campaigns/${id}/analytics`);
      const data = await response.json();
      setAnalytics(data);
    } catch (error) {
      console.error("Error fetching analytics:", error);
    }
  };

  const handleLaunch = async () => {
    try {
      await fetch(`http://localhost:8000/v1/marketing/campaigns/${id}/launch`, {
        method: "POST",
      });
      fetchCampaign();
    } catch (error) {
      console.error("Error launching campaign:", error);
    }
  };

  const handlePause = async () => {
    try {
      await fetch(`http://localhost:8000/v1/marketing/campaigns/${id}/pause`, {
        method: "POST",
      });
      fetchCampaign();
    } catch (error) {
      console.error("Error pausing campaign:", error);
    }
  };

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to delete this campaign?")) return;

    try {
      await fetch(`http://localhost:8000/v1/marketing/campaigns/${id}`, {
        method: "DELETE",
      });
      router.push("/dashboard/marketing/campaigns");
    } catch (error) {
      console.error("Error deleting campaign:", error);
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
            <p className="mt-4 text-slate-600">Loading campaign...</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  if (!campaign) {
    return (
      <DashboardLayout>
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold text-slate-900 mb-4">Campaign not found</h2>
          <button
            onClick={() => router.push("/dashboard/marketing/campaigns")}
            className="text-blue-600 hover:text-blue-700 font-semibold"
          >
            ← Back to Campaigns
          </button>
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
            <button
              onClick={() => router.push("/dashboard/marketing/campaigns")}
              className="text-blue-600 hover:text-blue-700 font-semibold mb-2"
            >
              ← Back to Campaigns
            </button>
            <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-2">
              {getCampaignTypeIcon(campaign.campaign_type)} {campaign.name}
            </h1>
            <p className="text-slate-600 mt-1">{campaign.description}</p>
          </div>
          <div className="flex gap-2">
            {campaign.status === "draft" && (
              <button
                onClick={handleLaunch}
                className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
              >
                🚀 Launch Campaign
              </button>
            )}
            {campaign.status === "active" && (
              <button
                onClick={handlePause}
                className="bg-yellow-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-yellow-700 transition-colors"
              >
                ⏸️ Pause Campaign
              </button>
            )}
            {campaign.status === "paused" && (
              <button
                onClick={handleLaunch}
                className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
              >
                ▶️ Resume Campaign
              </button>
            )}
            <button
              onClick={handleDelete}
              className="bg-red-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-red-700 transition-colors"
            >
              🗑️ Delete
            </button>
          </div>
        </div>

        {/* Campaign Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-sm text-slate-600 mb-1">Status</p>
            <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(campaign.status)}`}>
              {campaign.status.toUpperCase()}
            </span>
          </div>

          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-sm text-slate-600 mb-1">Campaign Type</p>
            <p className="font-bold text-slate-900 capitalize">{campaign.campaign_type.replace("_", " ")}</p>
          </div>

          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-sm text-slate-600 mb-1">Features</p>
            <div className="flex gap-1">
              {campaign.is_ab_test && (
                <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-semibold">
                  A/B Test
                </span>
              )}
              {campaign.is_automated && (
                <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-semibold">
                  Automated
                </span>
              )}
              {!campaign.is_ab_test && !campaign.is_automated && (
                <span className="text-slate-500 text-sm">None</span>
              )}
            </div>
          </div>

          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-sm text-slate-600 mb-1">Created</p>
            <p className="font-bold text-slate-900">{new Date(campaign.created_at).toLocaleDateString()}</p>
          </div>
        </div>

        {/* Analytics */}
        {analytics && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white shadow-md">
                <p className="text-sm opacity-90 font-medium">Total Sent</p>
                <p className="text-3xl font-bold mt-2">{analytics.total_sent.toLocaleString()}</p>
                <p className="text-xs opacity-75 mt-1">{analytics.delivery_rate.toFixed(1)}% delivered</p>
              </div>

              <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-6 text-white shadow-md">
                <p className="text-sm opacity-90 font-medium">Opened</p>
                <p className="text-3xl font-bold mt-2">{analytics.unique_opened.toLocaleString()}</p>
                <p className="text-xs opacity-75 mt-1">{analytics.open_rate.toFixed(1)}% open rate</p>
              </div>

              <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-6 text-white shadow-md">
                <p className="text-sm opacity-90 font-medium">Clicked</p>
                <p className="text-3xl font-bold mt-2">{analytics.unique_clicked.toLocaleString()}</p>
                <p className="text-xs opacity-75 mt-1">{analytics.click_rate.toFixed(1)}% click rate</p>
              </div>

              <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-lg p-6 text-white shadow-md">
                <p className="text-sm opacity-90 font-medium">Revenue</p>
                <p className="text-3xl font-bold mt-2">${analytics.total_revenue.toLocaleString()}</p>
                <p className="text-xs opacity-75 mt-1">{analytics.total_conversions} conversions</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
                <h3 className="font-bold text-slate-900 mb-4">Delivery Metrics</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Delivered</span>
                    <span className="font-bold text-green-600">{analytics.total_delivered.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Bounced</span>
                    <span className="font-bold text-yellow-600">{analytics.total_bounced.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Failed</span>
                    <span className="font-bold text-red-600">{analytics.total_failed.toLocaleString()}</span>
                  </div>
                </div>
              </div>

              <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
                <h3 className="font-bold text-slate-900 mb-4">Engagement Metrics</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-slate-600">Open Rate</span>
                      <span className="text-sm font-bold text-slate-900">{analytics.open_rate.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${analytics.open_rate}%` }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-slate-600">Click Rate</span>
                      <span className="text-sm font-bold text-slate-900">{analytics.click_rate.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full"
                        style={{ width: `${analytics.click_rate}%` }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-slate-600">Conversion Rate</span>
                      <span className="text-sm font-bold text-slate-900">{analytics.conversion_rate.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full"
                        style={{ width: `${analytics.conversion_rate}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Campaign Details */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <h3 className="font-bold text-slate-900 mb-4">Campaign Details</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-slate-600">Subject Line</p>
              <p className="font-semibold text-slate-900">{campaign.subject_line || "Not set"}</p>
            </div>
            <div>
              <p className="text-sm text-slate-600">Scheduled At</p>
              <p className="font-semibold text-slate-900">
                {campaign.scheduled_at
                  ? new Date(campaign.scheduled_at).toLocaleString()
                  : "Not scheduled"}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-600">Segment ID</p>
              <p className="font-semibold text-slate-900">{campaign.segment_id || "Not set"}</p>
            </div>
            <div>
              <p className="text-sm text-slate-600">Template ID</p>
              <p className="font-semibold text-slate-900">{campaign.template_id || "Not set"}</p>
            </div>
            <div>
              <p className="text-sm text-slate-600">Last Updated</p>
              <p className="font-semibold text-slate-900">{new Date(campaign.updated_at).toLocaleString()}</p>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

