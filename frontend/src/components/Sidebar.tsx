"use client";
import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface NavItem {
  id: string;
  label: string;
  icon: string;
  href: string;
  badge?: number;
  submenu?: NavItem[];
}

const navItems: NavItem[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    icon: "📊",
    href: "/dashboard",
  },
  {
    id: "leads",
    label: "Leads",
    icon: "👥",
    href: "/dashboard/leads",
    badge: 12,
    submenu: [
      { id: "leads-all", label: "All Leads", icon: "📋", href: "/dashboard/leads" },
      { id: "leads-new", label: "New Leads", icon: "✨", href: "/dashboard/leads?status=new" },
      { id: "leads-qualified", label: "Qualified", icon: "⭐", href: "/dashboard/leads?status=qualified" },
      { id: "leads-contacted", label: "Contacted", icon: "📞", href: "/dashboard/leads?status=contacted" },
    ],
  },
  {
    id: "customers",
    label: "Customers",
    icon: "🏢",
    href: "/dashboard/customers",
    badge: 48,
    submenu: [
      { id: "customers-all", label: "All Customers", icon: "📋", href: "/dashboard/customers" },
      { id: "customers-active", label: "Active", icon: "✅", href: "/dashboard/customers?status=active" },
      { id: "customers-inactive", label: "Inactive", icon: "⏸️", href: "/dashboard/customers?status=inactive" },
    ],
  },
  {
    id: "policies",
    label: "Policies",
    icon: "📄",
    href: "/dashboard/policies",
    badge: 156,
    submenu: [
      { id: "policies-all", label: "All Policies", icon: "📋", href: "/dashboard/policies" },
      { id: "policies-auto", label: "Auto Insurance", icon: "🚗", href: "/dashboard/policies?type=auto" },
      { id: "policies-home", label: "Home Insurance", icon: "🏠", href: "/dashboard/policies?type=home" },
      { id: "policies-life", label: "Life Insurance", icon: "❤️", href: "/dashboard/policies?type=life" },
      { id: "policies-health", label: "Health Insurance", icon: "⚕️", href: "/dashboard/policies?type=health" },
    ],
  },
  {
    id: "claims",
    label: "Claims",
    icon: "🔔",
    href: "/dashboard/claims",
    badge: 8,
    submenu: [
      { id: "claims-all", label: "All Claims", icon: "📋", href: "/dashboard/claims" },
      { id: "claims-pending", label: "Pending", icon: "⏳", href: "/dashboard/claims?status=pending" },
      { id: "claims-approved", label: "Approved", icon: "✅", href: "/dashboard/claims?status=approved" },
      { id: "claims-rejected", label: "Rejected", icon: "❌", href: "/dashboard/claims?status=rejected" },
    ],
  },
  {
    id: "documents",
    label: "Documents",
    icon: "📑",
    href: "/dashboard/documents",
    submenu: [
      { id: "docs-all", label: "All Documents", icon: "📋", href: "/dashboard/documents" },
      { id: "docs-pending", label: "Pending Signature", icon: "✍️", href: "/dashboard/documents?status=pending" },
      { id: "docs-signed", label: "Signed", icon: "✅", href: "/dashboard/documents?status=signed" },
      { id: "docs-templates", label: "Templates", icon: "📝", href: "/dashboard/documents/templates" },
    ],
  },
  {
    id: "communications",
    label: "Communications",
    icon: "💬",
    href: "/dashboard/communications",
    submenu: [
      { id: "comm-emails", label: "Emails", icon: "📧", href: "/dashboard/communications/emails" },
      { id: "comm-sms", label: "SMS", icon: "💬", href: "/dashboard/communications/sms" },
      { id: "comm-calls", label: "Call Logs", icon: "☎️", href: "/dashboard/communications/calls" },
      { id: "comm-campaigns", label: "Campaigns", icon: "📢", href: "/dashboard/communications/campaigns" },
    ],
  },
  {
    id: "reports",
    label: "Reports",
    icon: "📈",
    href: "/dashboard/reports",
    submenu: [
      { id: "reports-sales", label: "Sales Report", icon: "💰", href: "/dashboard/reports/sales" },
      { id: "reports-pipeline", label: "Pipeline Report", icon: "📊", href: "/dashboard/reports/pipeline" },
      { id: "reports-performance", label: "Performance", icon: "⚡", href: "/dashboard/reports/performance" },
      { id: "reports-analytics", label: "Analytics", icon: "📉", href: "/dashboard/analytics" },
    ],
  },
  {
    id: "settings",
    label: "Settings",
    icon: "⚙️",
    href: "/dashboard/settings",
    submenu: [
      { id: "settings-profile", label: "Profile", icon: "👤", href: "/dashboard/settings/profile" },
      { id: "settings-team", label: "Team", icon: "👥", href: "/dashboard/settings/team" },
      { id: "settings-integrations", label: "Integrations", icon: "🔗", href: "/dashboard/settings/integrations" },
      { id: "settings-notifications", label: "Notifications", icon: "🔔", href: "/dashboard/settings/notifications" },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [expandedItems, setExpandedItems] = useState<string[]>(["dashboard"]);

  const toggleExpand = (id: string) => {
    setExpandedItems((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const isActive = (href: string) => {
    return pathname === href || pathname.startsWith(href + "/");
  };

  return (
    <aside className="w-64 bg-gradient-to-b from-blue-900 to-blue-800 text-white shadow-lg h-screen overflow-y-auto sticky top-0">
      {/* Logo */}
      <div className="p-6 border-b border-blue-700">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🏢</span>
          <div>
            <h1 className="text-xl font-bold">LEAGAI</h1>
            <p className="text-xs text-blue-200">Insurance CRM</p>
          </div>
        </div>
      </div>

      {/* Navigation Items */}
      <nav className="p-4 space-y-2">
        {navItems.map((item) => (
          <div key={item.id}>
            {/* Main Item */}
            <div className="flex items-center">
              <Link
                href={item.href}
                className={`flex-1 flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                  isActive(item.href)
                    ? "bg-blue-600 text-white shadow-md"
                    : "text-blue-100 hover:bg-blue-700 hover:text-white"
                }`}
              >
                <span className="text-lg">{item.icon}</span>
                <span className="font-medium text-sm">{item.label}</span>
                {item.badge && (
                  <span className="ml-auto bg-red-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                    {item.badge}
                  </span>
                )}
              </Link>

              {/* Expand/Collapse Button */}
              {item.submenu && (
                <button
                  onClick={() => toggleExpand(item.id)}
                  className="px-2 py-3 text-blue-100 hover:text-white transition-all"
                >
                  <span className={`transition-transform ${expandedItems.includes(item.id) ? "rotate-180" : ""}`}>
                    ▼
                  </span>
                </button>
              )}
            </div>

            {/* Submenu */}
            {item.submenu && expandedItems.includes(item.id) && (
              <div className="ml-4 mt-2 space-y-1 border-l-2 border-blue-600 pl-2">
                {item.submenu.map((subitem) => (
                  <Link
                    key={subitem.id}
                    href={subitem.href}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all ${
                      isActive(subitem.href)
                        ? "bg-blue-600 text-white"
                        : "text-blue-100 hover:bg-blue-700 hover:text-white"
                    }`}
                  >
                    <span>{subitem.icon}</span>
                    <span>{subitem.label}</span>
                  </Link>
                ))}
              </div>
            )}
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-blue-700 bg-blue-900">
        <div className="text-xs text-blue-200 text-center">
          <p>© 2025 LEAGAI</p>
          <p>Insurance Lead Generation</p>
        </div>
      </div>
    </aside>
  );
}

