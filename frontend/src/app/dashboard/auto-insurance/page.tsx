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
            <p className="text-3xl font-bold text-blue-700 mt-2">12</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Coverage options</p>
          </div>
          <div className="bg-white border-2 border-emerald-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Categories</p>
            <p className="text-3xl font-bold text-emerald-700 mt-2">4</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Core, Personal, Specialized, Special Use</p>
          </div>
          <div className="bg-white border-2 border-purple-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-purple-700 mt-2">428</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Currently active</p>
          </div>
          <div className="bg-white border-2 border-amber-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Coverage</p>
            <p className="text-3xl font-bold text-amber-700 mt-2">$52.4M</p>
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
            <p className="text-3xl font-bold text-amber-700 mt-2">1,208</p>
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
              <p className="text-sm text-slate-600 mt-1">12 comprehensive coverage options across 4 categories</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-bold text-blue-700">4 Categories</p>
              <p className="text-xs text-slate-600">12 Products</p>
            </div>
          </div>

          {/* Category: Core Coverage */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-blue-600">●</span> Core Coverage (3)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Liability Coverage */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">⚖️</span>
                  <h3 className="font-bold text-slate-900">Liability Coverage</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Required in most states</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Bodily Injury Liability</li>
                  <li>• Property Damage Liability</li>
                  <li>• Legal defense costs</li>
                </ul>
              </div>

              {/* Collision Coverage */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">💥</span>
                  <h3 className="font-bold text-slate-900">Collision Coverage</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Covers vehicle damage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Accident damage repair</li>
                  <li>• Single-car accidents</li>
                  <li>• Deductible options</li>
                </ul>
              </div>

              {/* Comprehensive Coverage */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🛡️</span>
                  <h3 className="font-bold text-slate-900">Comprehensive Coverage</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Non-collision damage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Theft protection</li>
                  <li>• Weather damage</li>
                  <li>• Vandalism coverage</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Personal Protection */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-emerald-600">●</span> Personal Protection (3)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Personal Injury Protection (PIP) */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏥</span>
                  <h3 className="font-bold text-slate-900">Personal Injury Protection</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">No-fault coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Medical expenses</li>
                  <li>• Lost wages</li>
                  <li>• Funeral costs</li>
                </ul>
              </div>

              {/* Uninsured/Underinsured Motorist */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🚫</span>
                  <h3 className="font-bold text-slate-900">Uninsured Motorist</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Protection from uninsured drivers</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• UM bodily injury</li>
                  <li>• UM property damage</li>
                  <li>• Underinsured coverage</li>
                </ul>
              </div>

              {/* Medical Payments Coverage */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">💊</span>
                  <h3 className="font-bold text-slate-900">Medical Payments</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">MedPay coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Medical bills</li>
                  <li>• Hospital expenses</li>
                  <li>• Passenger coverage</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Specialized Coverage */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-purple-600">●</span> Specialized Coverage (4)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Gap Insurance */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">📉</span>
                  <h3 className="font-bold text-slate-900">Gap Insurance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Loan/lease protection</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Covers loan balance</li>
                  <li>• Total loss protection</li>
                  <li>• Depreciation coverage</li>
                </ul>
              </div>

              {/* Rental Reimbursement */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🔑</span>
                  <h3 className="font-bold text-slate-900">Rental Reimbursement</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Temporary transportation</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Rental car costs</li>
                  <li>• Daily limits</li>
                  <li>• Repair period coverage</li>
                </ul>
              </div>

              {/* Roadside Assistance */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🔧</span>
                  <h3 className="font-bold text-slate-900">Roadside Assistance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Emergency services</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Towing service</li>
                  <li>• Jump starts</li>
                  <li>• Flat tire assistance</li>
                </ul>
              </div>

              {/* Custom Equipment Coverage */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🎵</span>
                  <h3 className="font-bold text-slate-900">Custom Equipment</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Aftermarket additions</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Sound systems</li>
                  <li>• Custom wheels</li>
                  <li>• Performance parts</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Special Use Cases */}
          <div>
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-amber-600">●</span> Special Use Cases (2)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Rideshare Coverage */}
              <div className="border-2 border-amber-100 rounded-lg p-4 hover:border-amber-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🚕</span>
                  <h3 className="font-bold text-slate-900">Rideshare Coverage</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Uber/Lyft drivers</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Period 1 coverage</li>
                  <li>• Commercial use</li>
                  <li>• Gap protection</li>
                </ul>
              </div>

              {/* Classic Car Insurance */}
              <div className="border-2 border-amber-100 rounded-lg p-4 hover:border-amber-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏎️</span>
                  <h3 className="font-bold text-slate-900">Classic Car Insurance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Vintage vehicles</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Agreed value coverage</li>
                  <li>• Limited mileage</li>
                  <li>• Restoration coverage</li>
                </ul>
              </div>
            </div>
          </div>
        </div>


      </div>
    </DashboardLayout>
  );
}

