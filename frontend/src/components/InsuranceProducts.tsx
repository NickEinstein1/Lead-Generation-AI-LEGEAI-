"use client";
import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface Product {
  id: string;
  name: string;
  icon: string;
  leads: number;
  revenue: string;
  conversionRate: number;
  color: string;
}

interface InsuranceProductsProps {
  products?: Product[];
}

export default function InsuranceProducts({ products }: InsuranceProductsProps) {
  const router = useRouter();
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [dynamicProducts, setDynamicProducts] = useState<Product[]>([]);

  const handleViewDetails = (product: Product) => {
    // Navigate to the product-specific page
    router.push(`/dashboard/life-insurance/${product.id}`);
  };

  // Generate dynamic values on component mount
  useEffect(() => {
    const generateDynamicProducts = (): Product[] => {
      const randomInRange = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;
      const randomDecimal = (min: number, max: number) => (Math.random() * (max - min) + min).toFixed(1);

      return [
        {
          id: "auto",
          name: "Auto Insurance",
          icon: "🚗",
          leads: randomInRange(350, 420),
          revenue: `$${randomInRange(130, 160)},${randomInRange(100, 999).toString().padStart(3, '0')}`,
          conversionRate: parseFloat(randomDecimal(23, 30)),
          color: "from-blue-500 to-blue-600",
        },
        {
          id: "home",
          name: "Home Insurance",
          icon: "🏠",
          leads: randomInRange(220, 290),
          revenue: `$${randomInRange(85, 115)},${randomInRange(100, 999).toString().padStart(3, '0')}`,
          conversionRate: parseFloat(randomDecimal(25, 32)),
          color: "from-amber-500 to-amber-600",
        },
        {
          id: "life",
          name: "Life Insurance",
          icon: "❤️",
          leads: randomInRange(170, 230),
          revenue: `$${randomInRange(75, 100)},${randomInRange(100, 999).toString().padStart(3, '0')}`,
          conversionRate: parseFloat(randomDecimal(20, 26)),
          color: "from-red-500 to-red-600",
        },
        {
          id: "health",
          name: "Health Insurance",
          icon: "⚕️",
          leads: randomInRange(370, 450),
          revenue: `$${randomInRange(140, 175)},${randomInRange(100, 999).toString().padStart(3, '0')}`,
          conversionRate: parseFloat(randomDecimal(21, 28)),
          color: "from-green-500 to-green-600",
        },
      ];
    };

    setDynamicProducts(generateDynamicProducts());
  }, []); // Runs once on mount

  const insuranceProducts = products || dynamicProducts;

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <h2 className="text-xl font-bold text-slate-900 mb-6">Insurance Products</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {insuranceProducts.map((product) => (
          <div
            key={product.id}
            className={`bg-gradient-to-br ${product.color} rounded-lg p-4 text-white shadow-md hover:shadow-lg transition-shadow cursor-pointer`}
          >
            <div className="text-3xl mb-2">{product.icon}</div>
            <h3 className="font-bold text-sm mb-3">{product.name}</h3>
            
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="opacity-90">Leads:</span>
                <span className="font-bold">{product.leads}</span>
              </div>
              <div className="flex justify-between">
                <span className="opacity-90">Revenue:</span>
                <span className="font-bold">{product.revenue}</span>
              </div>
              <div className="flex justify-between">
                <span className="opacity-90">Conversion:</span>
                <span className="font-bold">{product.conversionRate}%</span>
              </div>
            </div>

            <button
              onClick={() => handleViewDetails(product)}
              className="w-full mt-3 bg-white bg-opacity-20 hover:bg-opacity-30 text-white font-semibold py-1 px-2 rounded text-xs transition-all active:scale-95"
            >
              View Details
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

