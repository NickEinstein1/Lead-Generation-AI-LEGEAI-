"use client";
import React, { useState, useEffect } from "react";

interface Metric {
  label: string;
  value: string | number;
  change: number;
  icon: string;
  color: string;
}

interface KeyMetricsProps {
  metrics?: Metric[];
}

export default function KeyMetrics({ metrics }: KeyMetricsProps) {
  const [dynamicMetrics, setDynamicMetrics] = useState<Metric[]>([]);

  // Generate dynamic values on component mount
  useEffect(() => {
    const generateDynamicMetrics = (): Metric[] => {
      const randomInRange = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;
      const randomDecimal = (min: number, max: number) => (Math.random() * (max - min) + min).toFixed(1);

      return [
        {
          label: "Total Leads",
          value: randomInRange(1150, 1350),
          change: parseFloat(randomDecimal(12, 25)),
          icon: "👥",
          color: "from-blue-500 to-blue-600",
        },
        {
          label: "This Month",
          value: randomInRange(280, 380),
          change: parseFloat(randomDecimal(10, 22)),
          icon: "📈",
          color: "from-green-500 to-green-600",
        },
        {
          label: "Conversion Rate",
          value: `${randomDecimal(20, 30)}%`,
          change: parseFloat(randomDecimal(3, 8)),
          icon: "🎯",
          color: "from-purple-500 to-purple-600",
        },
        {
          label: "Avg Deal Value",
          value: `$${randomInRange(3200, 4500).toLocaleString()}`,
          change: parseFloat(randomDecimal(5, 15)),
          icon: "💰",
          color: "from-amber-500 to-amber-600",
        },
      ];
    };

    setDynamicMetrics(generateDynamicMetrics());
  }, []); // Runs once on mount

  const keyMetrics = metrics || dynamicMetrics;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
      {keyMetrics.map((metric, idx) => (
        <div
          key={idx}
          className={`bg-gradient-to-br ${metric.color} rounded-lg p-4 sm:p-6 text-white shadow-md hover:shadow-lg transition-shadow`}
        >
          <div className="flex items-start justify-between mb-3 sm:mb-4">
            <div>
              <p className="text-xs sm:text-sm opacity-90 font-medium">{metric.label}</p>
              <p className="text-2xl sm:text-3xl font-bold mt-1 sm:mt-2">{metric.value}</p>
            </div>
            <span className="text-2xl sm:text-3xl">{metric.icon}</span>
          </div>

          <div className="flex items-center gap-1 text-xs sm:text-sm">
            <span className={metric.change >= 0 ? "text-green-200" : "text-red-200"}>
              {metric.change >= 0 ? "↑" : "↓"}
            </span>
            <span className="opacity-90">
              {Math.abs(metric.change)}% vs last month
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

