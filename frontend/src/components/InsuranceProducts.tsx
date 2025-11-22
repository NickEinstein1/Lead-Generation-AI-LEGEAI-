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
    // Navigate to the product-specific page based on product ID
    const routeMap: { [key: string]: string } = {
      'auto': '/dashboard/auto-insurance',
      'home': '/dashboard/home-insurance',
      'life': '/dashboard/life-insurance',
      'health': '/dashboard/health-insurance',
    };

    const route = routeMap[product.id] || `/dashboard/${product.id}-insurance`;
    router.push(route);
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
    <div className="bg-slate-900/60 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6 shadow-2xl hover:shadow-cyan-500/20 transition-all duration-300 relative overflow-hidden">
      {/* Background glow */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-orange-500/10 to-cyan-500/10"></div>

      <h2 className="text-xl font-bold text-blue-100 mb-6 flex items-center gap-2 relative z-10">
        <span className="text-2xl">🏢</span>
        <span className="bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">Insurance Products</span>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 relative z-10">
        {insuranceProducts.map((product) => (
          <div
            key={product.id}
            onClick={() => handleViewDetails(product)}
            className={`group relative bg-slate-800/60 backdrop-blur-xl rounded-2xl p-4 text-white shadow-2xl hover:shadow-cyan-500/30 transition-all duration-300 hover:-translate-y-2 cursor-pointer overflow-hidden border border-blue-500/30 hover:border-cyan-400/60 active:scale-95`}
          >
            {/* Gradient overlay */}
            <div className={`absolute inset-0 bg-gradient-to-br ${product.color} opacity-20 group-hover:opacity-30 transition-opacity duration-300`}></div>

            {/* Glow effect */}
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-orange-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

            {/* Content */}
            <div className="relative z-10">
              <div className="text-4xl mb-2 group-hover:scale-125 group-hover:rotate-12 transition-all duration-300 filter drop-shadow-lg">{product.icon}</div>
              <h3 className="font-bold text-sm mb-3 tracking-wide text-cyan-200">{product.name}</h3>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between bg-cyan-500/20 backdrop-blur-sm rounded-lg px-2 py-1.5 border border-cyan-400/30">
                  <span className="text-cyan-300 font-medium">Leads:</span>
                  <span className="font-bold text-white">{product.leads}</span>
                </div>
                <div className="flex justify-between bg-cyan-500/20 backdrop-blur-sm rounded-lg px-2 py-1.5 border border-cyan-400/30">
                  <span className="text-cyan-300 font-medium">Revenue:</span>
                  <span className="font-bold text-white">{product.revenue}</span>
                </div>
                <div className="flex justify-between bg-cyan-500/20 backdrop-blur-sm rounded-lg px-2 py-1.5 border border-cyan-400/30">
                  <span className="text-cyan-300 font-medium">Conversion:</span>
                  <span className="font-bold text-white">{product.conversionRate}%</span>
                </div>
              </div>

              <div className="w-full mt-3 bg-gradient-to-r from-cyan-500/30 to-blue-500/30 group-hover:from-cyan-500/50 group-hover:to-blue-500/50 backdrop-blur-sm text-white font-bold py-2 px-2 rounded-lg text-xs transition-all shadow-lg group-hover:shadow-cyan-500/50 border border-cyan-400/30 group-hover:border-cyan-300/50 text-center">
                View Details →
              </div>
            </div>

            {/* Decorative corner accent */}
            <div className="absolute -bottom-6 -right-6 w-20 h-20 bg-cyan-500/20 rounded-full blur-2xl group-hover:scale-150 transition-transform duration-500"></div>

            {/* Border glow */}
            <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 shadow-[inset_0_0_20px_rgba(34,211,238,0.2)]"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

