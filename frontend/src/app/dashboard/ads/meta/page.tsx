'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Loader2, TrendingUp, TrendingDown, DollarSign, Eye, MousePointer, Users } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/v1';

interface Campaign {
  id: string;
  name: string;
  status: string;
  objective: string;
  daily_budget?: number;
  lifetime_budget?: number;
}

interface CampaignInsights {
  impressions: number;
  clicks: number;
  spend: number;
  cpc: number;
  cpm: number;
  ctr: number;
  reach: number;
  frequency: number;
}

interface AdAccount {
  id: string;
  name: string;
}

export default function MetaAdsDashboard() {
  const [isLoading, setIsLoading] = useState(true);
  const [adAccounts, setAdAccounts] = useState<AdAccount[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string>('');
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [selectedCampaign, setSelectedCampaign] = useState<string>('');
  const [insights, setInsights] = useState<CampaignInsights | null>(null);
  const [accountInsights, setAccountInsights] = useState<CampaignInsights | null>(null);

  useEffect(() => {
    loadAdAccounts();
  }, []);

  useEffect(() => {
    if (selectedAccount) {
      loadCampaigns();
      loadAccountInsights();
    }
  }, [selectedAccount]);

  useEffect(() => {
    if (selectedCampaign) {
      loadCampaignInsights();
    }
  }, [selectedCampaign]);

  const loadAdAccounts = async () => {
    try {
      const response = await fetch(`${API_BASE}/integrations/meta/ad-accounts`, {
        headers: {
          'X-Session-ID': 'demo-session',
        },
      });
      const data = await response.json();
      setAdAccounts(data);
      if (data.length > 0) {
        setSelectedAccount(data[0].id);
      }
    } catch (err) {
      console.error('Error loading ad accounts:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadCampaigns = async () => {
    try {
      const response = await fetch(
        `${API_BASE}/integrations/meta/campaigns?ad_account_id=${selectedAccount}`,
        {
          headers: {
            'X-Session-ID': 'demo-session',
          },
        }
      );
      const data = await response.json();
      setCampaigns(data.campaigns || []);
    } catch (err) {
      console.error('Error loading campaigns:', err);
    }
  };

  const loadCampaignInsights = async () => {
    try {
      const response = await fetch(
        `${API_BASE}/integrations/meta/insights/campaign/${selectedCampaign}?date_preset=last_7d`,
        {
          headers: {
            'X-Session-ID': 'demo-session',
          },
        }
      );
      const data = await response.json();
      if (data.data && data.data.length > 0) {
        setInsights(data.data[0]);
      }
    } catch (err) {
      console.error('Error loading campaign insights:', err);
    }
  };

  const loadAccountInsights = async () => {
    try {
      const response = await fetch(
        `${API_BASE}/integrations/meta/insights/account/${selectedAccount}?date_preset=last_30d`,
        {
          headers: {
            'X-Session-ID': 'demo-session',
          },
        }
      );
      const data = await response.json();
      if (data.data && data.data.length > 0) {
        setAccountInsights(data.data[0]);
      }
    } catch (err) {
      console.error('Error loading account insights:', err);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US').format(value);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (adAccounts.length === 0) {
    return (
      <div className="p-6">
        <Card>
          <CardHeader>
            <CardTitle>No Ad Accounts Connected</CardTitle>
            <CardDescription>
              Please connect your Meta account in Settings → Integrations → Meta
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Meta Ads Dashboard</h1>
          <p className="text-gray-600 mt-2">Monitor your Facebook and Instagram ad performance</p>
        </div>
        <Button onClick={() => window.location.href = '/dashboard/settings/integrations/meta'}>
          Manage Integration
        </Button>
      </div>

      {/* Ad Account Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Ad Account</CardTitle>
        </CardHeader>
        <CardContent>
          <select
            className="w-full p-2 border rounded-lg"
            value={selectedAccount}
            onChange={(e) => setSelectedAccount(e.target.value)}
          >
            {adAccounts.map((account) => (
              <option key={account.id} value={account.id}>
                {account.name} ({account.id})
              </option>
            ))}
          </select>
        </CardContent>
      </Card>

      {/* Account-Level Metrics */}
      {accountInsights && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Spend (30d)</CardTitle>
              <DollarSign className="h-4 w-4 text-gray-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatCurrency(accountInsights.spend)}</div>
              <p className="text-xs text-gray-600 mt-1">Last 30 days</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Impressions</CardTitle>
              <Eye className="h-4 w-4 text-gray-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(accountInsights.impressions)}</div>
              <p className="text-xs text-gray-600 mt-1">Total views</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Clicks</CardTitle>
              <MousePointer className="h-4 w-4 text-gray-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(accountInsights.clicks)}</div>
              <p className="text-xs text-gray-600 mt-1">
                CTR: {(accountInsights.ctr * 100).toFixed(2)}%
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Reach</CardTitle>
              <Users className="h-4 w-4 text-gray-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(accountInsights.reach)}</div>
              <p className="text-xs text-gray-600 mt-1">Unique users</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Campaigns List */}
      <Card>
        <CardHeader>
          <CardTitle>Campaigns</CardTitle>
          <CardDescription>
            {campaigns.length} campaign{campaigns.length !== 1 ? 's' : ''} in this account
          </CardDescription>
        </CardHeader>
        <CardContent>
          {campaigns.length === 0 ? (
            <div className="text-center py-8 text-gray-600">
              No campaigns found. Create your first campaign to get started.
            </div>
          ) : (
            <div className="space-y-3">
              {campaigns.map((campaign) => (
                <div
                  key={campaign.id}
                  className={`flex items-center justify-between p-4 border rounded-lg cursor-pointer hover:bg-gray-50 ${
                    selectedCampaign === campaign.id ? 'border-blue-500 bg-blue-50' : ''
                  }`}
                  onClick={() => setSelectedCampaign(campaign.id)}
                >
                  <div className="flex-1">
                    <div className="font-medium">{campaign.name}</div>
                    <div className="text-sm text-gray-600">
                      {campaign.id} • {campaign.objective}
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    {campaign.daily_budget && (
                      <div className="text-sm text-gray-600">
                        Daily: {formatCurrency(campaign.daily_budget / 100)}
                      </div>
                    )}
                    <Badge
                      variant={
                        campaign.status === 'ACTIVE'
                          ? 'default'
                          : campaign.status === 'PAUSED'
                          ? 'secondary'
                          : 'outline'
                      }
                    >
                      {campaign.status}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Campaign Insights */}
      {selectedCampaign && insights && (
        <Card>
          <CardHeader>
            <CardTitle>Campaign Performance (Last 7 Days)</CardTitle>
            <CardDescription>
              Detailed metrics for selected campaign
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 border rounded-lg">
                <div className="text-sm text-gray-600">Spend</div>
                <div className="text-2xl font-bold mt-1">{formatCurrency(insights.spend)}</div>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="text-sm text-gray-600">Impressions</div>
                <div className="text-2xl font-bold mt-1">{formatNumber(insights.impressions)}</div>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="text-sm text-gray-600">Clicks</div>
                <div className="text-2xl font-bold mt-1">{formatNumber(insights.clicks)}</div>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="text-sm text-gray-600">CTR</div>
                <div className="text-2xl font-bold mt-1">{(insights.ctr * 100).toFixed(2)}%</div>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="text-sm text-gray-600">CPC</div>
                <div className="text-2xl font-bold mt-1">{formatCurrency(insights.cpc)}</div>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="text-sm text-gray-600">CPM</div>
                <div className="text-2xl font-bold mt-1">{formatCurrency(insights.cpm)}</div>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="text-sm text-gray-600">Reach</div>
                <div className="text-2xl font-bold mt-1">{formatNumber(insights.reach)}</div>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="text-sm text-gray-600">Frequency</div>
                <div className="text-2xl font-bold mt-1">{insights.frequency.toFixed(2)}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

