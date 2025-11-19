"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import { API_BASE } from "@/lib/api";
import Link from "next/link";

interface PolicyType {
  policy_type: string;
  display_name: string;
  category: string;
  description: string;
  age_range: { min: number; max: number };
  coverage_range: { min: number; max: number };
  features: {
    cash_value: boolean;
    investment_component: boolean;
    premium_flexibility: string;
  };
  best_for: string[];
  key_features: string[];
  underwriting_complexity: string;
}

interface CategoryData {
  category_name: string;
  products: PolicyType[];
}

export default function LifeInsurancePage() {
  const router = useRouter();
  const [policyData, setPolicyData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPolicyTypes = async () => {
      try {
        const response = await fetch(`${API_BASE}/life-insurance/policy-types`);
        const data = await response.json();
        setPolicyData(data);
      } catch (error) {
        console.error("Failed to fetch policy types:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchPolicyTypes();
  }, []);

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-lg text-slate-600">Loading life insurance products...</div>
        </div>
      </DashboardLayout>
    );
  }

  const getCategoryIcon = (category: string): string => {
    const icons: Record<string, string> = {
      term: "⏱️",
      permanent: "♾️",
      annuity: "💰",
      specialty: "⭐",
      hybrid: "🔄",
    };
    return icons[category] || "📋";
  };

  const getCategoryColor = (category: string): string => {
    const colors: Record<string, string> = {
      term: "bg-green-50 border-green-200",
      permanent: "bg-blue-50 border-blue-200",
      annuity: "bg-yellow-50 border-yellow-200",
      specialty: "bg-purple-50 border-purple-200",
      hybrid: "bg-orange-50 border-orange-200",
    };
    return colors[category] || "bg-gray-50 border-gray-200";
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Life Insurance Products</h1>
            <p className="text-slate-600 font-medium mt-1">
              Comprehensive life insurance policy types and categories
            </p>
          </div>
          <button
            onClick={() => router.push('/leads/new')}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + Score New Lead
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Product Types</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">{policyData?.total_products || 0}</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Across all categories</p>
          </div>
          <div className="bg-white border-2 border-green-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Categories</p>
            <p className="text-3xl font-bold text-green-700 mt-2">
              {policyData?.categories ? Object.keys(policyData.categories).length : 0}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">Product categories</p>
          </div>
          <div className="bg-white border-2 border-purple-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-purple-700 mt-2">156</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Currently active</p>
          </div>
          <div className="bg-white border-2 border-amber-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Coverage</p>
            <p className="text-3xl font-bold text-amber-700 mt-2">$45.2M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Combined value</p>
          </div>
        </div>

        {/* Category Summary */}
        {policyData?.category_summary && (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
            <h2 className="text-xl font-bold text-slate-900 mb-4">Category Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(policyData.category_summary).map(([category, description]: [string, any]) => (
                <div key={category} className={`border-2 rounded-lg p-4 ${getCategoryColor(category)}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-2xl">{getCategoryIcon(category)}</span>
                    <h3 className="font-bold text-slate-900 capitalize">{category}</h3>
                  </div>
                  <p className="text-sm text-slate-600">{description}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Product Categories */}
        {policyData?.categories && Object.entries(policyData.categories).map(([category, categoryData]: [string, any]) => (
          <div key={category} className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
            <div className="p-6 border-b border-blue-200 bg-gradient-to-r from-blue-50 to-white">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{getCategoryIcon(category)}</span>
                  <div>
                    <h2 className="text-xl font-bold text-slate-900">{categoryData.category_name}</h2>
                    <p className="text-sm text-slate-600">{categoryData.products?.length || 0} products available</p>
                  </div>
                </div>
                <Link
                  href={`/dashboard/life-insurance/${category}`}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all text-sm"
                >
                  View All →
                </Link>
              </div>
            </div>

            {/* Product Grid */}
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {categoryData.products?.slice(0, 3).map((product: PolicyType) => (
                  <div key={product.policy_type} className="border-2 border-slate-200 rounded-lg p-4 hover:border-blue-400 hover:shadow-md transition-all">
                    <h3 className="font-bold text-slate-900 mb-2">{product.display_name}</h3>
                    <p className="text-sm text-slate-600 mb-3">{product.description}</p>

                    <div className="space-y-2 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-slate-700">Age Range:</span>
                        <span className="text-slate-600">{product.age_range.min}-{product.age_range.max} years</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-slate-700">Coverage:</span>
                        <span className="text-slate-600">
                          ${(product.coverage_range.min / 1000).toFixed(0)}K - ${(product.coverage_range.max / 1000000).toFixed(1)}M
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-slate-700">Cash Value:</span>
                        <span className={`px-2 py-0.5 rounded ${product.features.cash_value ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}`}>
                          {product.features.cash_value ? 'Yes' : 'No'}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-slate-700">Investment:</span>
                        <span className={`px-2 py-0.5 rounded ${product.features.investment_component ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}`}>
                          {product.features.investment_component ? 'Yes' : 'No'}
                        </span>
                      </div>
                    </div>

                    <div className="mt-3 pt-3 border-t border-slate-200">
                      <p className="text-xs font-semibold text-slate-700 mb-1">Best For:</p>
                      <div className="flex flex-wrap gap-1">
                        {product.best_for?.slice(0, 2).map((item, idx) => (
                          <span key={idx} className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {categoryData.products?.length > 3 && (
                <div className="mt-4 text-center">
                  <Link
                    href={`/dashboard/life-insurance/${category}`}
                    className="text-blue-600 hover:text-blue-700 font-semibold text-sm"
                  >
                    View all {categoryData.products.length} {categoryData.category_name} products →
                  </Link>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </DashboardLayout>
  );
}


