"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";

interface MarketingStats {
  total_campaigns: number;
  active_campaigns: number;
  total_segments: number;
  total_templates: number;
  total_sent: number;
  total_opened: number;
  total_clicked: number;
  total_conversions: number;
  total_revenue: number;
  avg_open_rate: number;
  avg_click_rate: number;
  avg_conversion_rate: number;
}

interface Campaign {
  id: number;
  name: string;
  description: string;
  campaign_type: string;
  status: string;
  target_count: number;
  created_at: string;
}

export default function MarketingAutomationPage() {
  const router = useRouter();
  const [stats, setStats] = useState<MarketingStats | null>(null);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMarketingData();
  }, []);

  const fetchMarketingData = async () => {
    try {
      // Fetch overview stats
      const statsRes = await fetch("http://localhost:8000/v1/marketing/analytics/overview");
      const statsData = await statsRes.json();
      setStats(statsData);

      // Fetch recent campaigns
      const campaignsRes = await fetch("http://localhost:8000/v1/marketing/campaigns?limit=10");
      const campaignsData = await campaignsRes.json();
      setCampaigns(campaignsData);
    } catch (error) {
      console.error("Error fetching marketing data:", error);
    } finally {
      setLoading(false);
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
            <p className="mt-4 text-slate-600">Loading marketing data...</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-2">
              📢 Marketing Automation
            </h1>
            <p className="text-slate-600 mt-1">
              Create, manage, and automate marketing campaigns based on customer data and behavior
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

        {/* Quick Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white shadow-md">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm opacity-90 font-medium">Total Campaigns</p>
                <p className="text-3xl font-bold mt-2">{stats?.total_campaigns || 0}</p>
              </div>
              <span className="text-3xl">📊</span>
            </div>
            <p className="text-xs opacity-75">{stats?.active_campaigns || 0} active</p>
          </div>

          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-6 text-white shadow-md">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm opacity-90 font-medium">Total Sent</p>
                <p className="text-3xl font-bold mt-2">{stats?.total_sent?.toLocaleString() || 0}</p>
              </div>
              <span className="text-3xl">📤</span>
            </div>
            <p className="text-xs opacity-75">Across all campaigns</p>
          </div>

          <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-6 text-white shadow-md">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm opacity-90 font-medium">Avg Open Rate</p>
                <p className="text-3xl font-bold mt-2">{stats?.avg_open_rate?.toFixed(1) || 0}%</p>
              </div>
              <span className="text-3xl">👁️</span>
            </div>
            <p className="text-xs opacity-75">{stats?.total_opened?.toLocaleString() || 0} total opens</p>
          </div>

          <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-lg p-6 text-white shadow-md">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm opacity-90 font-medium">Total Revenue</p>
                <p className="text-3xl font-bold mt-2">${stats?.total_revenue?.toLocaleString() || 0}</p>
              </div>
              <span className="text-3xl">💰</span>
            </div>
            <p className="text-xs opacity-75">{stats?.total_conversions || 0} conversions</p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <button
            onClick={() => router.push("/dashboard/marketing/campaigns")}
            className="bg-white border-2 border-blue-200 rounded-lg p-6 hover:shadow-lg transition-all text-left group"
          >
            <div className="text-4xl mb-3">📧</div>
            <h3 className="font-bold text-slate-900 mb-1 group-hover:text-blue-600">Campaigns</h3>
            <p className="text-sm text-slate-600">Manage all campaigns</p>
          </button>

          <button
            onClick={() => router.push("/dashboard/marketing/segments")}
            className="bg-white border-2 border-blue-200 rounded-lg p-6 hover:shadow-lg transition-all text-left group"
          >
            <div className="text-4xl mb-3">👥</div>
            <h3 className="font-bold text-slate-900 mb-1 group-hover:text-blue-600">Segments</h3>
            <p className="text-sm text-slate-600">Define audience segments</p>
          </button>

          <button
            onClick={() => router.push("/dashboard/marketing/templates")}
            className="bg-white border-2 border-blue-200 rounded-lg p-6 hover:shadow-lg transition-all text-left group"
          >
            <div className="text-4xl mb-3">📝</div>
            <h3 className="font-bold text-slate-900 mb-1 group-hover:text-blue-600">Templates</h3>
            <p className="text-sm text-slate-600">Email & SMS templates</p>
          </button>

          <button
            onClick={() => router.push("/dashboard/marketing/automation")}
            className="bg-white border-2 border-blue-200 rounded-lg p-6 hover:shadow-lg transition-all text-left group"
          >
            <div className="text-4xl mb-3">⚡</div>
            <h3 className="font-bold text-slate-900 mb-1 group-hover:text-blue-600">Automation</h3>
            <p className="text-sm text-slate-600">Triggers & workflows</p>
          </button>
        </div>

        {/* Recent Campaigns */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-slate-900">Recent Campaigns</h2>
            <button
              onClick={() => router.push("/dashboard/marketing/campaigns")}
              className="text-blue-600 hover:text-blue-700 font-semibold text-sm"
            >
              View All →
            </button>
          </div>

          {campaigns.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📢</div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">No campaigns yet</h3>
              <p className="text-slate-600 mb-4">Create your first marketing campaign to get started</p>
              <button
                onClick={() => router.push("/dashboard/marketing/campaigns/create")}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                Create Campaign
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b-2 border-slate-200">
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Campaign</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Type</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Status</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Target</th>
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
                      <td className="py-4 px-4 text-slate-700">{campaign.target_count.toLocaleString()}</td>
                      <td className="py-4 px-4 text-slate-600 text-sm">
                        {new Date(campaign.created_at).toLocaleDateString()}
                      </td>
                      <td className="py-4 px-4 text-right">
                        <button
                          onClick={() => router.push(`/dashboard/marketing/campaigns/${campaign.id}`)}
                          className="text-blue-600 hover:text-blue-700 font-semibold text-sm"
                        >
                          View →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
            <h3 className="font-bold text-slate-900 mb-4">Engagement Metrics</h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-600">Open Rate</span>
                  <span className="text-sm font-bold text-slate-900">{stats?.avg_open_rate?.toFixed(1) || 0}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${stats?.avg_open_rate || 0}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-600">Click Rate</span>
                  <span className="text-sm font-bold text-slate-900">{stats?.avg_click_rate?.toFixed(1) || 0}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div
                    className="bg-green-600 h-2 rounded-full"
                    style={{ width: `${stats?.avg_click_rate || 0}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-600">Conversion Rate</span>
                  <span className="text-sm font-bold text-slate-900">{stats?.avg_conversion_rate?.toFixed(1) || 0}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div
                    className="bg-purple-600 h-2 rounded-full"
                    style={{ width: `${stats?.avg_conversion_rate || 0}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
            <h3 className="font-bold text-slate-900 mb-4">Resources</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-slate-700">Active Segments</span>
                <span className="font-bold text-blue-600">{stats?.total_segments || 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-700">Templates</span>
                <span className="font-bold text-blue-600">{stats?.total_templates || 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-700">Active Campaigns</span>
                <span className="font-bold text-green-600">{stats?.active_campaigns || 0}</span>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-blue-200 rounded-lg p-6 shadow-md">
            <h3 className="font-bold text-slate-900 mb-4">💡 Quick Tips</h3>
            <ul className="space-y-2 text-sm text-slate-700">
              <li className="flex items-start gap-2">
                <span>✓</span>
                <span>Segment your audience for better targeting</span>
              </li>
              <li className="flex items-start gap-2">
                <span>✓</span>
                <span>A/B test subject lines to improve open rates</span>
              </li>
              <li className="flex items-start gap-2">
                <span>✓</span>
                <span>Use automation triggers for timely engagement</span>
              </li>
              <li className="flex items-start gap-2">
                <span>✓</span>
                <span>Monitor analytics to optimize performance</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
