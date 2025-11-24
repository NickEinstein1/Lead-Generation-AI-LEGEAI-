"use client";
import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
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
  const router = useRouter();
  const params = useParams();
  const category = params.category as string;
  const [products, setProducts] = useState<PolicyType[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showQuoteModal, setShowQuoteModal] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<PolicyType | null>(null);
  const [quoteFormData, setQuoteFormData] = useState({
    name: "",
    email: "",
    phone: "",
    age: "",
    coverage_amount: "",
    health_status: "good",
    smoking_status: "non_smoker"
  });

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

  const handleGetQuote = (product: PolicyType) => {
    setSelectedProduct(product);
    setShowQuoteModal(true);
  };

  const handleSubmitQuote = async () => {
    if (!quoteFormData.name || !quoteFormData.email || !quoteFormData.age || !quoteFormData.coverage_amount) {
      alert("Please fill in all required fields");
      return;
    }

    try {
      // Generate unique idempotency key
      const idempotencyKey = `life-quote-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

      // Split name into first and last name
      const nameParts = quoteFormData.name.trim().split(' ');
      const firstName = nameParts[0] || '';
      const lastName = nameParts.slice(1).join(' ') || '';

      // Prepare lead data for API
      const leadData = {
        idempotency_key: idempotencyKey,
        source: "life_insurance_quote",
        channel: "web",
        product_interest: "life",
        contact: {
          first_name: firstName,
          last_name: lastName,
          email: quoteFormData.email,
          phone: quoteFormData.phone || null
        },
        attributes: {
          age: parseInt(quoteFormData.age),
          coverage_amount: parseFloat(quoteFormData.coverage_amount),
          health_status: quoteFormData.health_status,
          smoking_status: quoteFormData.smoking_status,
          product_type: selectedProduct?.type || "unknown",
          product_name: selectedProduct?.display_name || "Unknown Product",
          category: params.category
        },
        consent: {
          marketing: true,
          terms_accepted: true,
          timestamp: new Date().toISOString()
        }
      };

      // Submit to backend API
      const response = await fetch("http://localhost:8000/v1/leads", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(leadData),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();

      // Show success message with lead ID
      alert(`✅ Quote request submitted successfully!\n\n` +
            `Lead ID: ${result.lead_id}\n` +
            `Product: ${selectedProduct?.display_name}\n` +
            `Name: ${quoteFormData.name}\n` +
            `Coverage: $${quoteFormData.coverage_amount}\n\n` +
            `Our team will contact you within 24 hours.`);

      // Reset form and close modal
      setShowQuoteModal(false);
      setQuoteFormData({
        name: "",
        email: "",
        phone: "",
        age: "",
        coverage_amount: "",
        health_status: "good",
        smoking_status: "non_smoker"
      });
      setSelectedProduct(null);
    } catch (error) {
      console.error("Failed to submit quote:", error);
      alert("❌ Failed to submit quote request. Please try again.\n\nError: " + (error instanceof Error ? error.message : "Unknown error"));
    }
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
          <button
            onClick={() => router.push('/leads/new')}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
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
                <button
                  onClick={() => handleGetQuote(product)}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all text-sm active:scale-95"
                >
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

        {/* Get Quote Modal */}
        {showQuoteModal && selectedProduct && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowQuoteModal(false)}>
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-2xl font-bold text-slate-900 mb-2">💼 Get Quote</h3>
              <p className="text-sm text-slate-600 mb-4">Request a quote for <span className="font-semibold text-blue-600">{selectedProduct.display_name}</span></p>

              <div className="space-y-4">
                {/* Personal Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Full Name *</label>
                    <input
                      type="text"
                      value={quoteFormData.name}
                      onChange={(e) => setQuoteFormData({ ...quoteFormData, name: e.target.value })}
                      placeholder="John Smith"
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Email Address *</label>
                    <input
                      type="email"
                      value={quoteFormData.email}
                      onChange={(e) => setQuoteFormData({ ...quoteFormData, email: e.target.value })}
                      placeholder="john@example.com"
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Phone Number</label>
                    <input
                      type="tel"
                      value={quoteFormData.phone}
                      onChange={(e) => setQuoteFormData({ ...quoteFormData, phone: e.target.value })}
                      placeholder="(555) 123-4567"
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Age *</label>
                    <input
                      type="number"
                      value={quoteFormData.age}
                      onChange={(e) => setQuoteFormData({ ...quoteFormData, age: e.target.value })}
                      placeholder="35"
                      min={selectedProduct.age_range.min}
                      max={selectedProduct.age_range.max}
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                    <p className="text-xs text-slate-500 mt-1">Age range: {selectedProduct.age_range.min}-{selectedProduct.age_range.max} years</p>
                  </div>
                </div>

                {/* Coverage Information */}
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Desired Coverage Amount *</label>
                  <input
                    type="number"
                    value={quoteFormData.coverage_amount}
                    onChange={(e) => setQuoteFormData({ ...quoteFormData, coverage_amount: e.target.value })}
                    placeholder="500000"
                    min={selectedProduct.coverage_range.min}
                    max={selectedProduct.coverage_range.max}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Coverage range: ${(selectedProduct.coverage_range.min / 1000).toFixed(0)}K - ${(selectedProduct.coverage_range.max / 1000000).toFixed(1)}M
                  </p>
                </div>

                {/* Health Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Health Status</label>
                    <select
                      value={quoteFormData.health_status}
                      onChange={(e) => setQuoteFormData({ ...quoteFormData, health_status: e.target.value })}
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    >
                      <option value="excellent">Excellent</option>
                      <option value="good">Good</option>
                      <option value="fair">Fair</option>
                      <option value="poor">Poor</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Smoking Status</label>
                    <select
                      value={quoteFormData.smoking_status}
                      onChange={(e) => setQuoteFormData({ ...quoteFormData, smoking_status: e.target.value })}
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    >
                      <option value="non_smoker">Non-Smoker</option>
                      <option value="former_smoker">Former Smoker</option>
                      <option value="smoker">Smoker</option>
                    </select>
                  </div>
                </div>

                {/* Product Summary */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-sm font-semibold text-slate-900 mb-2">Selected Product Summary:</p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-slate-600">Product:</span>
                      <span className="ml-2 font-semibold text-slate-900">{selectedProduct.display_name}</span>
                    </div>
                    <div>
                      <span className="text-slate-600">Category:</span>
                      <span className="ml-2 font-semibold text-slate-900 capitalize">{selectedProduct.category}</span>
                    </div>
                    <div>
                      <span className="text-slate-600">Premium Type:</span>
                      <span className="ml-2 font-semibold text-slate-900 capitalize">{selectedProduct.features.premium_flexibility}</span>
                    </div>
                    <div>
                      <span className="text-slate-600">Underwriting:</span>
                      <span className="ml-2 font-semibold text-slate-900 capitalize">{selectedProduct.underwriting_complexity}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowQuoteModal(false);
                    setQuoteFormData({
                      name: "",
                      email: "",
                      phone: "",
                      age: "",
                      coverage_amount: "",
                      health_status: "good",
                      smoking_status: "non_smoker"
                    });
                    setSelectedProduct(null);
                  }}
                  className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSubmitQuote}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                >
                  Submit Quote Request
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}


