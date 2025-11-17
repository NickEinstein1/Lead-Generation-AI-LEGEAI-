"use client";
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { API_BASE } from "@/lib/api";

interface NavItem {
  id: string;
  label: string;
  icon: string;
  href: string;
  badge?: number;
  submenu?: NavItem[];
}

interface PolicyType {
  policy_type: string;
  display_name: string;
  category: string;
  description: string;
}

const getNavItems = (lifeInsuranceSubmenu: NavItem[]): NavItem[] => [
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
    id: "life-insurance",
    label: "Life Insurance",
    icon: "❤️",
    href: "/dashboard/life-insurance",
    badge: 35,
    submenu: lifeInsuranceSubmenu,
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
    label: "E-Signatures",
    icon: "✍️",
    href: "/dashboard/documents",
    submenu: [
      { id: "docs-all", label: "All Documents", icon: "📋", href: "/dashboard/documents" },
      { id: "docs-pending", label: "Pending Signature", icon: "⏳", href: "/dashboard/documents?status=pending" },
      { id: "docs-signed", label: "Signed", icon: "✅", href: "/dashboard/documents?status=signed" },
      { id: "docs-templates", label: "Templates", icon: "📝", href: "/dashboard/documents/templates" },
    ],
  },
  {
    id: "file-library",
    label: "File Library",
    icon: "📁",
    href: "/dashboard/file-library",
    submenu: [
      { id: "files-all", label: "All Files", icon: "📋", href: "/dashboard/file-library" },
      { id: "files-policies", label: "Policies", icon: "📄", href: "/dashboard/file-library?category=policies" },
      { id: "files-claims", label: "Claims", icon: "📝", href: "/dashboard/file-library?category=claims" },
      { id: "files-customer-data", label: "Customer Data", icon: "👥", href: "/dashboard/file-library?category=customer_data" },
      { id: "files-reports", label: "Reports", icon: "📊", href: "/dashboard/file-library?category=reports" },
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

// Helper function to get icon for policy category
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

export default function Sidebar() {
  const pathname = usePathname();
  const [expandedItems, setExpandedItems] = useState<string[]>(["dashboard"]);
  const [lifeInsuranceSubmenu, setLifeInsuranceSubmenu] = useState<NavItem[]>([]);
  const [loadingPolicyTypes, setLoadingPolicyTypes] = useState(true);

  // Fetch life insurance policy types from backend
  useEffect(() => {
    const fetchPolicyTypes = async () => {
      try {
        const response = await fetch(`${API_BASE}/life-insurance/policy-types`);
        const data = await response.json();

        // Build submenu from categories
        const submenu: NavItem[] = [
          {
            id: "life-all",
            label: "All Life Insurance",
            icon: "📋",
            href: "/dashboard/life-insurance"
          }
        ];

        // Add category-based submenus
        if (data.categories) {
          Object.entries(data.categories).forEach(([category, categoryData]: [string, any]) => {
            const categoryIcon = getCategoryIcon(category);
            submenu.push({
              id: `life-${category}`,
              label: categoryData.category_name || category,
              icon: categoryIcon,
              href: `/dashboard/life-insurance/${category}`,
            });
          });
        }

        setLifeInsuranceSubmenu(submenu);
      } catch (error) {
        console.error("Failed to fetch life insurance policy types:", error);
        // Fallback submenu
        setLifeInsuranceSubmenu([
          { id: "life-all", label: "All Life Insurance", icon: "📋", href: "/dashboard/life-insurance" },
          { id: "life-term", label: "Term Life", icon: "⏱️", href: "/dashboard/life-insurance/term" },
          { id: "life-permanent", label: "Permanent Life", icon: "♾️", href: "/dashboard/life-insurance/permanent" },
          { id: "life-annuity", label: "Annuities", icon: "💰", href: "/dashboard/life-insurance/annuity" },
          { id: "life-specialty", label: "Specialty Products", icon: "⭐", href: "/dashboard/life-insurance/specialty" },
        ]);
      } finally {
        setLoadingPolicyTypes(false);
      }
    };

    fetchPolicyTypes();
  }, []);

  const toggleExpand = (id: string) => {
    setExpandedItems((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const isActive = (href: string) => {
    return pathname === href || pathname.startsWith(href + "/");
  };

  // Get navigation items with dynamic life insurance submenu
  const navItems = getNavItems(lifeInsuranceSubmenu);

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
        {loadingPolicyTypes && navItems.length === 0 ? (
          <div className="text-center text-blue-200 py-4">Loading...</div>
        ) : (
          navItems.map((item) => (
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
          ))
        )}
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

