"use client";
import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
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

export default function CategoryPage() {
  const params = useParams();
  const category = params.category as string;
  const [products, setProducts] = useState<PolicyType[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCategoryProducts = async () => {
      try {
        const response = await fetch(`${API_BASE}/life-insurance/policy-types/${category}`);
        if (!response.ok) {
          throw new Error("Category not found");
        }
        const data = await response.json();
        setProducts(data.products || []);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (category) {
      fetchCategoryProducts();
    }
  }, [category]);

  const getCategoryIcon = (cat: string): string => {
    const icons: Record<string, string> = {
      term: "⏱️",
      permanent: "♾️",
      annuity: "💰",
      specialty: "⭐",
      hybrid: "🔄",
    };
    return icons[cat] || "📋";
  };

  const getComplexityColor = (complexity: string): string => {
    const colors: Record<string, string> = {
      simple: "bg-green-100 text-green-700",
      moderate: "bg-yellow-100 text-yellow-700",
      complex: "bg-red-100 text-red-700",
    };
    return colors[complexity] || "bg-gray-100 text-gray-700";
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-lg text-slate-600">Loading products...</div>
        </div>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout>
        <div className="flex flex-col items-center justify-center h-64">
          <div className="text-lg text-red-600 mb-4">Error: {error}</div>
          <Link href="/dashboard/life-insurance" className="text-blue-600 hover:text-blue-700 underline">
            ← Back to Life Insurance
          </Link>
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
            <div className="flex items-center gap-3 mb-2">
              <Link href="/dashboard/life-insurance" className="text-blue-600 hover:text-blue-700">
                ← Back
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-4xl">{getCategoryIcon(category)}</span>
              <div>
                <h1 className="text-3xl font-bold text-slate-900 capitalize">{category} Life Insurance</h1>
                <p className="text-slate-600 font-medium mt-1">{products.length} products available</p>
              </div>
            </div>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Score New Lead
          </button>
        </div>

        {/* Products Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {products.map((product) => (
            <div key={product.policy_type} className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg transition-all">
              {/* Product Header */}
              <div className="mb-4">
                <h2 className="text-xl font-bold text-slate-900 mb-2">{product.display_name}</h2>
                <p className="text-sm text-slate-600">{product.description}</p>
              </div>

              {/* Product Details */}
              <div className="space-y-3 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold text-slate-700">Age Range:</span>
                  <span className="text-slate-600">{product.age_range.min}-{product.age_range.max} years</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold text-slate-700">Coverage:</span>
                  <span className="text-slate-600">
                    ${(product.coverage_range.min / 1000).toFixed(0)}K - ${(product.coverage_range.max / 1000000).toFixed(1)}M
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold text-slate-700">Premium Type:</span>
                  <span className="text-slate-600 capitalize">{product.features.premium_flexibility}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold text-slate-700">Underwriting:</span>
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${getComplexityColor(product.underwriting_complexity)}`}>
                    {product.underwriting_complexity}
                  </span>
                </div>
              </div>

              {/* Features */}
              <div className="mb-4">
                <div className="flex gap-2 mb-2">
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${product.features.cash_value ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}`}>
                    {product.features.cash_value ? '✓ Cash Value' : '✗ No Cash Value'}
                  </span>
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${product.features.investment_component ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}`}>
                    {product.features.investment_component ? '✓ Investment' : '✗ No Investment'}
                  </span>
                </div>
              </div>

              {/* Best For */}
              <div className="mb-4 pb-4 border-b border-slate-200">
                <p className="text-xs font-semibold text-slate-700 mb-2">Best For:</p>
                <div className="flex flex-wrap gap-1">
                  {product.best_for?.map((item, idx) => (
                    <span key={idx} className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded">
                      {item}
                    </span>
                  ))}
                </div>
              </div>

              {/* Key Features */}
              <div>
                <p className="text-xs font-semibold text-slate-700 mb-2">Key Features:</p>
                <ul className="space-y-1">
                  {product.key_features?.slice(0, 3).map((feature, idx) => (
                    <li key={idx} className="text-xs text-slate-600 flex items-start gap-2">
                      <span className="text-blue-600 mt-0.5">•</span>
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Action Button */}
              <div className="mt-4 pt-4 border-t border-slate-200">
                <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all text-sm">
                  Get Quote
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Empty State */}
        {products.length === 0 && (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-12 text-center">
            <p className="text-lg text-slate-600">No products found in this category.</p>
            <Link href="/dashboard/life-insurance" className="text-blue-600 hover:text-blue-700 underline mt-2 inline-block">
              View all categories
            </Link>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}


