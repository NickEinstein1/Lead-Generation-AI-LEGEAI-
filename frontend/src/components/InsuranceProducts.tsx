"use client";
import React from "react";

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
  const defaultProducts: Product[] = [
    {
      id: "auto",
      name: "Auto Insurance",
      icon: "🚗",
      leads: 156,
      revenue: "$45,200",
      conversionRate: 18.5,
      color: "from-blue-500 to-blue-600",
    },
    {
      id: "home",
      name: "Home Insurance",
      icon: "🏠",
      leads: 98,
      revenue: "$32,100",
      conversionRate: 22.3,
      color: "from-amber-500 to-amber-600",
    },
    {
      id: "life",
      name: "Life Insurance",
      icon: "❤️",
      leads: 67,
      revenue: "$28,500",
      conversionRate: 19.8,
      color: "from-red-500 to-red-600",
    },
    {
      id: "health",
      name: "Health Insurance",
      icon: "⚕️",
      leads: 124,
      revenue: "$38,900",
      conversionRate: 21.1,
      color: "from-green-500 to-green-600",
    },
  ];

  const insuranceProducts = products || defaultProducts;

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

            <button className="w-full mt-3 bg-white bg-opacity-20 hover:bg-opacity-30 text-white font-semibold py-1 px-2 rounded text-xs transition-all">
              View Details
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

