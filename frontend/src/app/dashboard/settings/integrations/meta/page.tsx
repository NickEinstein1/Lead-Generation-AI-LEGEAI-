'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Loader2, CheckCircle2, XCircle, ExternalLink, RefreshCw } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/v1';

interface AdAccount {
  id: string;
  name: string;
  account_status: string;
  currency: string;
  timezone_name: string;
}

interface Page {
  id: string;
  name: string;
  category: string;
  access_token?: string;
}

interface LeadForm {
  id: string;
  name: string;
  status: string;
  leads_count: number;
}

export default function MetaIntegrationPage() {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [adAccounts, setAdAccounts] = useState<AdAccount[]>([]);
  const [pages, setPages] = useState<Page[]>([]);
  const [selectedPage, setSelectedPage] = useState<string>('');
  const [leadForms, setLeadForms] = useState<LeadForm[]>([]);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    checkConnectionStatus();
  }, []);

  const checkConnectionStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/integrations/meta/status`, {
        headers: {
          'X-Session-ID': 'demo-session',
        },
      });
      const data = await response.json();
      setIsConnected(data.connected);
      
      if (data.connected) {
        loadAdAccounts();
        loadPages();
      }
    } catch (err) {
      console.error('Error checking status:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConnect = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE}/integrations/meta/auth/url`, {
        headers: {
          'X-Session-ID': 'demo-session',
        },
      });
      const data = await response.json();
      
      // Redirect to Meta authorization
      window.location.href = data.authorization_url;
    } catch (err) {
      setError('Failed to initiate connection');
      setIsLoading(false);
    }
  };

  const handleDisconnect = async () => {
    try {
      setIsLoading(true);
      await fetch(`${API_BASE}/integrations/meta/disconnect`, {
        method: 'POST',
        headers: {
          'X-Session-ID': 'demo-session',
        },
      });
      setIsConnected(false);
      setAdAccounts([]);
      setPages([]);
      setLeadForms([]);
    } catch (err) {
      setError('Failed to disconnect');
    } finally {
      setIsLoading(false);
    }
  };

  const loadAdAccounts = async () => {
    try {
      const response = await fetch(`${API_BASE}/integrations/meta/ad-accounts`, {
        headers: {
          'X-Session-ID': 'demo-session',
        },
      });
      const data = await response.json();
      setAdAccounts(data);
    } catch (err) {
      console.error('Error loading ad accounts:', err);
    }
  };

  const loadPages = async () => {
    try {
      const response = await fetch(`${API_BASE}/integrations/meta/pages`, {
        headers: {
          'X-Session-ID': 'demo-session',
        },
      });
      const data = await response.json();
      setPages(data);
    } catch (err) {
      console.error('Error loading pages:', err);
    }
  };

  const loadLeadForms = async (pageId: string) => {
    try {
      const response = await fetch(`${API_BASE}/integrations/meta/lead-forms?page_id=${pageId}`, {
        headers: {
          'X-Session-ID': 'demo-session',
        },
      });
      const data = await response.json();
      setLeadForms(data.forms || []);
      setSelectedPage(pageId);
    } catch (err) {
      console.error('Error loading lead forms:', err);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Meta Marketing API Integration</h1>
        <p className="text-gray-600 mt-2">
          Connect your Meta Business account to sync leads from Facebook and Instagram Lead Ads
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Connection Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Connection Status
            {isConnected ? (
              <CheckCircle2 className="h-5 w-5 text-green-600" />
            ) : (
              <XCircle className="h-5 w-5 text-gray-400" />
            )}
          </CardTitle>
          <CardDescription>
            {isConnected
              ? 'Your Meta account is connected and ready to sync leads'
              : 'Connect your Meta account to start syncing leads from Facebook and Instagram'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!isConnected ? (
            <Button onClick={handleConnect} disabled={isLoading}>
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Connect Meta Account
            </Button>
          ) : (
            <div className="flex gap-4">
              <Button variant="outline" onClick={checkConnectionStatus}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
              <Button variant="destructive" onClick={handleDisconnect}>
                Disconnect Account
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Ad Accounts */}
      {isConnected && adAccounts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Ad Accounts</CardTitle>
            <CardDescription>
              {adAccounts.length} ad account{adAccounts.length !== 1 ? 's' : ''} connected
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {adAccounts.map((account) => (
                <div
                  key={account.id}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div>
                    <div className="font-medium">{account.name}</div>
                    <div className="text-sm text-gray-600">
                      {account.id} • {account.currency} • {account.timezone_name}
                    </div>
                  </div>
                  <Badge variant={account.account_status === 'ACTIVE' ? 'default' : 'secondary'}>
                    {account.account_status}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pages */}
      {isConnected && pages.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Facebook Pages</CardTitle>
            <CardDescription>
              {pages.length} page{pages.length !== 1 ? 's' : ''} connected
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {pages.map((page) => (
                <div
                  key={page.id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 cursor-pointer"
                  onClick={() => loadLeadForms(page.id)}
                >
                  <div>
                    <div className="font-medium">{page.name}</div>
                    <div className="text-sm text-gray-600">
                      {page.id} • {page.category}
                    </div>
                  </div>
                  <Button variant="outline" size="sm">
                    View Lead Forms
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Lead Forms */}
      {selectedPage && leadForms.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Lead Forms</CardTitle>
            <CardDescription>
              {leadForms.length} lead form{leadForms.length !== 1 ? 's' : ''} found
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {leadForms.map((form) => (
                <div
                  key={form.id}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div>
                    <div className="font-medium">{form.name}</div>
                    <div className="text-sm text-gray-600">
                      {form.id} • {form.leads_count} leads
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={form.status === 'ACTIVE' ? 'default' : 'secondary'}>
                      {form.status}
                    </Badge>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
                        try {
                          const response = await fetch(`${API_BASE}/integrations/meta/sync-leads`, {
                            method: 'POST',
                            headers: {
                              'Content-Type': 'application/json',
                              'X-Session-ID': 'demo-session',
                            },
                            body: JSON.stringify({
                              form_id: form.id,
                              page_id: selectedPage,
                              limit: 100,
                            }),
                          });
                          const data = await response.json();
                          alert(`Synced ${data.synced} leads (${data.duplicates} duplicates)`);
                        } catch (err) {
                          alert('Failed to sync leads');
                        }
                      }}
                    >
                      Sync Leads
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Help Section */}
      <Card>
        <CardHeader>
          <CardTitle>Need Help?</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <ExternalLink className="h-4 w-4" />
            <a
              href="https://developers.facebook.com/docs/marketing-api"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              Meta Marketing API Documentation
            </a>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <ExternalLink className="h-4 w-4" />
            <a
              href="https://developers.facebook.com/docs/marketing-api/guides/lead-ads"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              Lead Ads Guide
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

