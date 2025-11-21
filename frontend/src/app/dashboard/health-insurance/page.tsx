"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import { API_BASE } from "@/lib/api";
import Link from "next/link";

export default function HealthInsurancePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE}/health-insurance/health`);
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error("Failed to fetch health insurance stats:", error);
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
          <div className="text-lg text-slate-600">Loading health insurance data...</div>
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
              <span className="text-4xl">⚕️</span>
              <div>
                <h1 className="text-3xl font-bold text-slate-900">Health Insurance</h1>
                <p className="text-slate-600 font-medium mt-1">
                  AI-powered lead scoring and analytics
                </p>
              </div>
            </div>
          </div>
          <Link
            href="/dashboard/health-insurance/score"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + Score New Lead
          </Link>
        </div>

        {/* Stats - Row 1 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Product Types</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">16</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Coverage options</p>
          </div>
          <div className="bg-white border-2 border-emerald-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Categories</p>
            <p className="text-3xl font-bold text-emerald-700 mt-2">5</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Major, Government, Supplemental, Specialty</p>
          </div>
          <div className="bg-white border-2 border-purple-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-purple-700 mt-2">1,847</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Currently active</p>
          </div>
          <div className="bg-white border-2 border-amber-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Coverage</p>
            <p className="text-3xl font-bold text-amber-700 mt-2">$94.2M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Annual premiums</p>
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
            <p className="text-3xl font-bold text-amber-700 mt-2">1,284</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Leads processed</p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href="/dashboard/health-insurance/score"
            className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg transition-all hover:border-blue-400"
          >
            <div className="flex items-center gap-4">
              <span className="text-4xl">⭐</span>
              <div>
                <h3 className="text-lg font-bold text-slate-900">Score Lead</h3>
                <p className="text-sm text-slate-600">Score individual health insurance leads</p>
              </div>
            </div>
          </Link>

          <Link
            href="/dashboard/health-insurance/analytics"
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

        {/* Health Insurance Product Types */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-bold text-slate-900">⚕️ Health Insurance Product Types</h2>
              <p className="text-sm text-slate-600 mt-1">16 comprehensive coverage options across 5 categories</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-bold text-blue-700">5 Categories</p>
              <p className="text-xs text-slate-600">16 Products</p>
            </div>
          </div>

          {/* Category: Major Medical Plans */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-blue-600">●</span> Major Medical Plans (5)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* HMO - Health Maintenance Organization */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏥</span>
                  <h3 className="font-bold text-slate-900">HMO Plans</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Health Maintenance Organization</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Primary care physician required</li>
                  <li>• Network-only coverage</li>
                  <li>• Lower premiums</li>
                </ul>
              </div>

              {/* PPO - Preferred Provider Organization */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏨</span>
                  <h3 className="font-bold text-slate-900">PPO Plans</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Preferred Provider Organization</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• No referrals needed</li>
                  <li>• Out-of-network coverage</li>
                  <li>• Higher flexibility</li>
                </ul>
              </div>

              {/* EPO - Exclusive Provider Organization */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏩</span>
                  <h3 className="font-bold text-slate-900">EPO Plans</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Exclusive Provider Organization</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Network providers only</li>
                  <li>• No referrals required</li>
                  <li>• Moderate premiums</li>
                </ul>
              </div>

              {/* POS - Point of Service */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🎯</span>
                  <h3 className="font-bold text-slate-900">POS Plans</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Point of Service</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• PCP coordination</li>
                  <li>• Out-of-network option</li>
                  <li>• Referral-based specialists</li>
                </ul>
              </div>

              {/* HDHP - High Deductible Health Plan */}
              <div className="border-2 border-blue-100 rounded-lg p-4 hover:border-blue-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">💰</span>
                  <h3 className="font-bold text-slate-900">HDHP Plans</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">High Deductible Health Plan</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• HSA eligible</li>
                  <li>• Lower monthly premiums</li>
                  <li>• Higher deductibles</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Government Programs */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-emerald-600">●</span> Government Programs (4)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Medicare */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">👴</span>
                  <h3 className="font-bold text-slate-900">Medicare</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Age 65+ or disabled</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Part A: Hospital insurance</li>
                  <li>• Part B: Medical insurance</li>
                  <li>• Part D: Prescription drugs</li>
                </ul>
              </div>

              {/* Medicare Advantage */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">⭐</span>
                  <h3 className="font-bold text-slate-900">Medicare Advantage</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Part C - All-in-one</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Private plan alternative</li>
                  <li>• Additional benefits</li>
                  <li>• Prescription coverage</li>
                </ul>
              </div>

              {/* Medigap */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🔗</span>
                  <h3 className="font-bold text-slate-900">Medigap</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Medicare Supplement</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Fills Medicare gaps</li>
                  <li>• Copays and deductibles</li>
                  <li>• Standardized plans A-N</li>
                </ul>
              </div>

              {/* Medicaid */}
              <div className="border-2 border-emerald-100 rounded-lg p-4 hover:border-emerald-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🤝</span>
                  <h3 className="font-bold text-slate-900">Medicaid</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Low-income assistance</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• State-federal program</li>
                  <li>• Income-based eligibility</li>
                  <li>• Comprehensive coverage</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Supplemental Coverage */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-purple-600">●</span> Supplemental Coverage (4)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

              {/* Dental Insurance */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🦷</span>
                  <h3 className="font-bold text-slate-900">Dental Insurance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Oral health coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Preventive care</li>
                  <li>• Basic procedures</li>
                  <li>• Major services</li>
                </ul>
              </div>

              {/* Vision Insurance */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">👓</span>
                  <h3 className="font-bold text-slate-900">Vision Insurance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Eye care coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Eye exams</li>
                  <li>• Glasses and contacts</li>
                  <li>• Corrective surgery discounts</li>
                </ul>
              </div>

              {/* Critical Illness Insurance */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">💔</span>
                  <h3 className="font-bold text-slate-900">Critical Illness</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Serious condition coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Lump-sum payment</li>
                  <li>• Cancer, heart attack, stroke</li>
                  <li>• Supplement to major medical</li>
                </ul>
              </div>

              {/* Hospital Indemnity */}
              <div className="border-2 border-purple-100 rounded-lg p-4 hover:border-purple-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🏥</span>
                  <h3 className="font-bold text-slate-900">Hospital Indemnity</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Hospitalization benefits</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Daily cash benefits</li>
                  <li>• Admission payments</li>
                  <li>• Covers out-of-pocket costs</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Specialty Plans */}
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-amber-600">●</span> Specialty Plans (2)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Catastrophic Health Insurance */}
              <div className="border-2 border-amber-100 rounded-lg p-4 hover:border-amber-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🚨</span>
                  <h3 className="font-bold text-slate-900">Catastrophic Plans</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Emergency coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Under 30 or hardship</li>
                  <li>• Very low premiums</li>
                  <li>• Major medical events</li>
                </ul>
              </div>

              {/* Short-Term Health Insurance */}
              <div className="border-2 border-amber-100 rounded-lg p-4 hover:border-amber-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">⏱️</span>
                  <h3 className="font-bold text-slate-900">Short-Term Plans</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Temporary coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Gap coverage</li>
                  <li>• 1-12 month terms</li>
                  <li>• Limited benefits</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Category: Accident & Injury */}
          <div>
            <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="text-red-600">●</span> Accident & Injury (1)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Accident Insurance */}
              <div className="border-2 border-red-100 rounded-lg p-4 hover:border-red-300 transition-all">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🚑</span>
                  <h3 className="font-bold text-slate-900">Accident Insurance</h3>
                </div>
                <p className="text-xs text-slate-600 mb-2">Injury-specific coverage</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• Emergency room visits</li>
                  <li>• Fractures and dislocations</li>
                  <li>• Ambulance services</li>
                </ul>
              </div>
            </div>
          </div>
        </div>


      </div>
    </DashboardLayout>
  );
}

