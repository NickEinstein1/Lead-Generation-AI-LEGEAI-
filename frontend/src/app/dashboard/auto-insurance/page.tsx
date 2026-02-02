"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import { API_BASE } from "@/lib/api";
import Link from "next/link";

export default function AutoInsurancePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE}/auto-insurance/health`);
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error("Failed to fetch auto insurance stats:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-lg text-slate-600">Loading auto insurance data...</div>
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
            <div className="flex items-center gap-3">
              <span className="text-4xl">🚗</span>
              <div>
                <h1 className="text-3xl font-bold text-slate-900">Auto Insurance</h1>
                <p className="text-slate-600 font-medium mt-1">
                  AI-powered lead scoring and analytics
                </p>
              </div>
            </div>
          </div>
          <Link
            href="/dashboard/auto-insurance/score"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + Score New Lead
          </Link>
        </div>

        {/* Stats - Row 1 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Product Types</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Coverage options</p>
          </div>
          <div className="bg-white border-2 border-emerald-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Categories</p>
            <p className="text-3xl font-bold text-emerald-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Catalog not available</p>
          </div>
          <div className="bg-white border-2 border-purple-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-purple-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Currently active</p>
          </div>
          <div className="bg-white border-2 border-amber-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Coverage</p>
            <p className="text-3xl font-bold text-amber-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Combined value</p>
          </div>
        </div>

        {/* Stats - Row 2: System Performance */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">System Status</p>
            <p className="text-3xl font-bold text-green-600 mt-2">
              {stats?.status === 'healthy' ? '✓ Healthy' : '✗ Error'}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">All systems operational</p>
          </div>
          <div className="bg-white border-2 border-emerald-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Scoring Engine</p>
            <p className="text-3xl font-bold text-emerald-700 mt-2">
              {stats?.xgboost_available ? '✓ Active' : '✗ Inactive'}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">AI scoring ready</p>
          </div>
          <div className="bg-white border-2 border-purple-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Analytics Engine</p>
            <p className="text-3xl font-bold text-purple-700 mt-2">
              {stats?.ensemble_available ? '✓ Active' : '✗ Inactive'}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">Data processing ready</p>
          </div>
          <div className="bg-white border-2 border-amber-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Leads</p>
            <p className="text-3xl font-bold text-amber-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Leads processed</p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href="/dashboard/auto-insurance/score"
            className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg transition-all hover:border-blue-400"
          >
            <div className="flex items-center gap-4">
              <span className="text-4xl">⭐</span>
              <div>
                <h3 className="text-lg font-bold text-slate-900">Score Lead</h3>
                <p className="text-sm text-slate-600">Score individual auto insurance leads</p>
              </div>
            </div>
          </Link>

          <Link
            href="/dashboard/auto-insurance/analytics"
            className="bg-white border-2 border-emerald-200 rounded-lg p-6 shadow-md hover:shadow-lg transition-all hover:border-emerald-400"
          >
            <div className="flex items-center gap-4">
              <span className="text-4xl">📈</span>
              <div>
                <h3 className="text-lg font-bold text-slate-900">Analytics</h3>
                <p className="text-sm text-slate-600">View performance metrics</p>
              </div>
            </div>
          </Link>
        </div>

        {/* Auto Insurance Product Types */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-bold text-slate-900">🚗 Auto Insurance Product Types</h2>
              <p className="text-sm text-slate-600 mt-1">Product catalog data not available.</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-bold text-blue-700">-</p>
              <p className="text-xs text-slate-600">No catalog data</p>
            </div>
          </div>
          <div className="rounded-lg border border-blue-100 bg-blue-50 p-4 text-sm text-slate-600">
            Product catalog is not available yet.
          </div>
        </div>


      </div>
    </DashboardLayout>
  );
}


