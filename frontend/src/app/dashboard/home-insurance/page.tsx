"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import { API_BASE } from "@/lib/api";
import Link from "next/link";

export default function HomeInsurancePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE}/home-insurance/health`);
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error("Failed to fetch home insurance stats:", error);
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
          <div className="text-lg text-slate-600">Loading home insurance data...</div>
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
              <span className="text-4xl">🏠</span>
              <div>
                <h1 className="text-3xl font-bold text-slate-900">Home Insurance</h1>
                <p className="text-slate-600 font-medium mt-1">
                  AI-powered lead scoring and analytics
                </p>
              </div>
            </div>
          </div>
          <Link
            href="/dashboard/home-insurance/score"
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
            <p className="text-xs text-slate-600 font-medium mt-2">No catalog data</p>
          </div>
          <div className="bg-white border-2 border-emerald-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Categories</p>
            <p className="text-3xl font-bold text-emerald-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No category data</p>
          </div>
          <div className="bg-white border-2 border-purple-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-purple-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No policy data</p>
          </div>
          <div className="bg-white border-2 border-amber-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Coverage</p>
            <p className="text-3xl font-bold text-amber-700 mt-2">-</p>
            <p className="text-xs text-slate-600 font-medium mt-2">No coverage data</p>
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
            <p className="text-xs text-slate-600 font-medium mt-2">No lead data</p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href="/dashboard/home-insurance/score"
            className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg transition-all hover:border-blue-400"
          >
            <div className="flex items-center gap-4">
              <span className="text-4xl">⭐</span>
              <div>
                <h3 className="text-lg font-bold text-slate-900">Score Lead</h3>
                <p className="text-sm text-slate-600">Score individual home insurance leads</p>
              </div>
            </div>
          </Link>

          <Link
            href="/dashboard/home-insurance/analytics"
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

        {/* Home Insurance Product Types */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-bold text-slate-900">🏠 Home Insurance Product Types</h2>
              <p className="text-sm text-slate-600 mt-1">12 comprehensive coverage options across 4 categories</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-bold text-blue-700">4 Categories</p>
              <p className="text-xs text-slate-600">12 Products</p>
            </div>
          </div>

          {/* Category: Standard Homeowners Policies */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-blue-600">●</span> Standard Homeowners Policies (5)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* HO-3 Standard Homeowners */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏡</span>
                  <h3 className="font-bold text-slate-900">HO-3 Standard Policy</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Most common homeowners policy</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Dwelling coverage</li>
                  <li>• Personal property</li>
                  <li>• Liability protection</li>
                </ul>
              </div>

              {/* HO-5 Comprehensive */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏰</span>
                  <h3 className="font-bold text-slate-900">HO-5 Comprehensive</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Premium open-peril coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• All-risk dwelling</li>
                  <li>• All-risk personal property</li>
                  <li>• Higher coverage limits</li>
                </ul>
              </div>

              {/* HO-4 Renters Insurance */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🔑</span>
                  <h3 className="font-bold text-slate-900">HO-4 Renters</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">For tenants and renters</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Personal property</li>
                  <li>• Liability coverage</li>
                  <li>• Additional living expenses</li>
                </ul>
              </div>

              {/* HO-6 Condo Insurance */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏢</span>
                  <h3 className="font-bold text-slate-900">HO-6 Condo</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Condominium owners</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Interior coverage</li>
                  <li>• Loss assessment</li>
                  <li>• Personal belongings</li>
                </ul>
              </div>

              {/* HO-8 Older Homes */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏚️</span>
                  <h3 className="font-bold text-slate-900">HO-8 Older Homes</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Historic or older properties</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Actual cash value</li>
                  <li>• Cost-effective coverage</li>
                  <li>• Named perils</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Specialized Property Coverage */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-emerald-600">●</span> Specialized Property Coverage (3)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Umbrella Policy */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">☂️</span>
                  <h3 className="font-bold text-slate-900">Umbrella Policy</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Excess liability coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• $1M+ liability limits</li>
                  <li>• Legal defense costs</li>
                  <li>• Multi-policy coverage</li>
                </ul>
              </div>

              {/* Dwelling Fire Policy */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🔥</span>
                  <h3 className="font-bold text-slate-900">Dwelling Fire Policy</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Rental or vacant properties</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Investment properties</li>
                  <li>• Fire and lightning</li>
                  <li>• Basic named perils</li>
                </ul>
              </div>

              {/* Manufactured Home */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏕️</span>
                  <h3 className="font-bold text-slate-900">Manufactured Home</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Mobile/modular homes</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Specialized coverage</li>
                  <li>• Transportation damage</li>
                  <li>• Tie-down equipment</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Natural Disaster Coverage */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-purple-600">●</span> Natural Disaster Coverage (2)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Flood Insurance */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🌊</span>
                  <h3 className="font-bold text-slate-900">Flood Insurance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">NFIP or private flood</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Rising water damage</li>
                  <li>• Building coverage</li>
                  <li>• Contents coverage</li>
                </ul>
              </div>

              {/* Earthquake Insurance */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🌋</span>
                  <h3 className="font-bold text-slate-900">Earthquake Insurance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Seismic activity protection</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Structural damage</li>
                  <li>• Personal property</li>
                  <li>• Additional living expenses</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Premium & Specialty */}
          <div>
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-amber-600">●</span> Premium & Specialty (2)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Luxury Home Insurance */}
              <div className="border-2 border-amber-100 rounded-lg p-4 hover:border-amber-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">💎</span>
                  <h3 className="font-bold text-slate-900">Luxury Home Insurance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">High-value properties</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Extended replacement cost</li>
                  <li>• Valuable items coverage</li>
                  <li>• Concierge services</li>
                </ul>
              </div>

              {/* Vacation Home Insurance */}
              <div className="border-2 border-amber-100 rounded-lg p-4 hover:border-amber-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏖️</span>
                  <h3 className="font-bold text-slate-900">Vacation Home</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Secondary residences</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Seasonal property</li>
                  <li>• Vacancy coverage</li>
                  <li>• Rental income protection</li>
                </ul>
              </div>
            </div>
          </div>
        </div>


      </div>
    </DashboardLayout>
  );
}

