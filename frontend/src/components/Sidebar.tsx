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

const getNavItems = (
  lifeInsuranceSubmenu: NavItem[],
  autoInsuranceSubmenu: NavItem[],
  homeInsuranceSubmenu: NavItem[],
  healthInsuranceSubmenu: NavItem[]
): NavItem[] => [
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
    id: "auto-insurance",
    label: "Auto Insurance",
    icon: "🚗",
    href: "/dashboard/auto-insurance",
    badge: 28,
    submenu: autoInsuranceSubmenu,
  },
  {
    id: "home-insurance",
    label: "Home Insurance",
    icon: "🏠",
    href: "/dashboard/home-insurance",
    badge: 22,
    submenu: homeInsuranceSubmenu,
  },
  {
    id: "health-insurance",
    label: "Health Insurance",
    icon: "⚕️",
    href: "/dashboard/health-insurance",
    badge: 31,
    submenu: healthInsuranceSubmenu,
  },
  {
    id: "customers",
    label: "Customers",
    icon: "🏢",
    href: "/dashboard/customers",
    badge: 48,
    submenu: [
      { id: "customers-all", label: "All Customers", icon: "📋", href: "/dashboard/customers" },
      { id: "customers-active", label: "Active", icon: "✅", href: "/dashboard/customers/active" },
      { id: "customers-inactive", label: "Inactive", icon: "⏸️", href: "/dashboard/customers/inactive" },
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
      { id: "policies-auto", label: "Auto Insurance", icon: "🚗", href: "/dashboard/policies/auto" },
      { id: "policies-home", label: "Home Insurance", icon: "🏠", href: "/dashboard/policies/home" },
      { id: "policies-life", label: "Life Insurance", icon: "❤️", href: "/dashboard/policies/life" },
      { id: "policies-health", label: "Health Insurance", icon: "⚕️", href: "/dashboard/policies/health" },
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
      { id: "claims-pending", label: "Pending", icon: "⏳", href: "/dashboard/claims/pending" },
      { id: "claims-approved", label: "Approved", icon: "✅", href: "/dashboard/claims/approved" },
      { id: "claims-rejected", label: "Rejected", icon: "❌", href: "/dashboard/claims/rejected" },
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
    id: "scheduler",
    label: "Scheduler",
    icon: "📅",
    href: "/dashboard/scheduler",
    submenu: [
      { id: "scheduler-all", label: "All Meetings", icon: "📋", href: "/dashboard/scheduler" },
      { id: "scheduler-create", label: "Schedule Meeting", icon: "➕", href: "/dashboard/scheduler/create" },
      { id: "scheduler-today", label: "Today's Meetings", icon: "📅", href: "/dashboard/scheduler?view=today" },
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
  const [autoInsuranceSubmenu, setAutoInsuranceSubmenu] = useState<NavItem[]>([]);
  const [homeInsuranceSubmenu, setHomeInsuranceSubmenu] = useState<NavItem[]>([]);
  const [healthInsuranceSubmenu, setHealthInsuranceSubmenu] = useState<NavItem[]>([]);
  const [loadingPolicyTypes, setLoadingPolicyTypes] = useState(true);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  // Keyboard shortcut to toggle sidebar (Ctrl/Cmd + B)
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        setIsOpen(prev => !prev);
      }
      // ESC to close sidebar
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isOpen]);

  // Fetch all insurance types from backend
  useEffect(() => {
    const fetchAllInsuranceTypes = async () => {
      try {
        // Fetch Life Insurance
        const lifeResponse = await fetch(`${API_BASE}/life-insurance/policy-types`);
        const lifeData = await lifeResponse.json();
        const lifeSubmenu: NavItem[] = [
          { id: "life-all", label: "All Life Insurance", icon: "📋", href: "/dashboard/life-insurance" }
        ];
        if (lifeData.categories) {
          Object.entries(lifeData.categories).forEach(([category, categoryData]: [string, any]) => {
            const categoryIcon = getCategoryIcon(category);
            lifeSubmenu.push({
              id: `life-${category}`,
              label: categoryData.category_name || category,
              icon: categoryIcon,
              href: `/dashboard/life-insurance/${category}`,
            });
          });
        }
        setLifeInsuranceSubmenu(lifeSubmenu);

        // Auto Insurance submenu
        setAutoInsuranceSubmenu([
          { id: "auto-all", label: "All Auto Insurance", icon: "📋", href: "/dashboard/auto-insurance" },
          { id: "auto-score", label: "Score Lead", icon: "⭐", href: "/dashboard/auto-insurance/score" },
          { id: "auto-analytics", label: "Analytics", icon: "📈", href: "/dashboard/auto-insurance/analytics" },
        ]);

        // Home Insurance submenu
        setHomeInsuranceSubmenu([
          { id: "home-all", label: "All Home Insurance", icon: "📋", href: "/dashboard/home-insurance" },
          { id: "home-score", label: "Score Lead", icon: "⭐", href: "/dashboard/home-insurance/score" },
          { id: "home-analytics", label: "Analytics", icon: "📈", href: "/dashboard/home-insurance/analytics" },
        ]);

        // Health Insurance submenu
        setHealthInsuranceSubmenu([
          { id: "health-all", label: "All Health Insurance", icon: "📋", href: "/dashboard/health-insurance" },
          { id: "health-score", label: "Score Lead", icon: "⭐", href: "/dashboard/health-insurance/score" },
          { id: "health-analytics", label: "Analytics", icon: "📈", href: "/dashboard/health-insurance/analytics" },
        ]);

      } catch (error) {
        console.error("Failed to fetch insurance types:", error);
        // Fallback submenus
        setLifeInsuranceSubmenu([
          { id: "life-all", label: "All Life Insurance", icon: "📋", href: "/dashboard/life-insurance" },
          { id: "life-term", label: "Term Life", icon: "⏱️", href: "/dashboard/life-insurance/term" },
          { id: "life-permanent", label: "Permanent Life", icon: "♾️", href: "/dashboard/life-insurance/permanent" },
          { id: "life-annuity", label: "Annuities", icon: "💰", href: "/dashboard/life-insurance/annuity" },
          { id: "life-specialty", label: "Specialty Products", icon: "⭐", href: "/dashboard/life-insurance/specialty" },
        ]);
        setAutoInsuranceSubmenu([
          { id: "auto-all", label: "All Auto Insurance", icon: "📋", href: "/dashboard/auto-insurance" },
        ]);
        setHomeInsuranceSubmenu([
          { id: "home-all", label: "All Home Insurance", icon: "📋", href: "/dashboard/home-insurance" },
        ]);
        setHealthInsuranceSubmenu([
          { id: "health-all", label: "All Health Insurance", icon: "📋", href: "/dashboard/health-insurance" },
        ]);
      } finally {
        setLoadingPolicyTypes(false);
      }
    };

    fetchAllInsuranceTypes();
  }, []);

  const toggleExpand = (id: string) => {
    setExpandedItems((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const isActive = (href: string) => {
    return pathname === href || pathname.startsWith(href + "/");
  };

  // Auto-close sidebar when clicking a link (optional, for better UX)
  const handleLinkClick = () => {
    // Close sidebar on mobile/tablet after clicking a link
    if (window.innerWidth < 1024) {
      setIsOpen(false);
    }
  };

  // Get navigation items with dynamic insurance submenus
  const navItems = getNavItems(lifeInsuranceSubmenu, autoInsuranceSubmenu, homeInsuranceSubmenu, healthInsuranceSubmenu);

  return (
    <>
      {/* Hamburger Menu Button - Responsive positioning */}
      <div className={`fixed top-4 sm:top-6 z-50 group transition-all duration-300 ${
        isOpen ? 'left-[272px]' : 'left-4 sm:left-6'
      }`}>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-2.5 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110 relative"
          aria-label="Toggle Sidebar"
          title="Toggle Sidebar (Ctrl+B)"
        >
          <div className="w-5 h-4 flex flex-col justify-between">
            <span className={`block h-0.5 w-full bg-white rounded transition-all duration-300 ${
              isOpen ? 'rotate-45 translate-y-1.5' : ''
            }`}></span>
            <span className={`block h-0.5 w-full bg-white rounded transition-all duration-300 ${
              isOpen ? 'opacity-0' : ''
            }`}></span>
            <span className={`block h-0.5 w-full bg-white rounded transition-all duration-300 ${
              isOpen ? '-rotate-45 -translate-y-1.5' : ''
            }`}></span>
          </div>

          {/* Pulse animation when closed */}
          {!isOpen && (
            <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500"></span>
            </span>
          )}
        </button>

        {/* Tooltip */}
        <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2.5 py-1.5 bg-gray-900 text-white text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap pointer-events-none shadow-xl">
          {isOpen ? 'Close Menu (Esc)' : 'Open Menu (Ctrl+B)'}
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900"></div>
        </div>
      </div>

      {/* Sidebar - Responsive width */}
      <aside className={`fixed top-0 left-0 w-64 sm:w-64 md:w-72 lg:w-64 bg-gradient-to-b from-blue-900 via-blue-800 to-blue-900 text-white shadow-2xl h-screen overflow-y-auto border-r-2 border-blue-700 scrollbar-thin scrollbar-thumb-blue-600 scrollbar-track-blue-900 z-40 transition-transform duration-300 ease-in-out ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* Logo */}
        <div className="p-5 border-b-2 border-blue-700 bg-gradient-to-r from-blue-800 to-blue-900">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 group cursor-pointer">
              <span className="text-2xl transform group-hover:scale-110 transition-transform duration-300">🏢</span>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
                  LEGEAI
                </h1>
                <p className="text-[10px] text-blue-300 font-medium">Insurance CRM</p>
              </div>
            </div>
            {/* Close button */}
            <button
              onClick={() => setIsOpen(false)}
              className="p-1.5 hover:bg-blue-700 rounded-lg transition-colors ml-2"
              aria-label="Close Sidebar"
            >
              <span className="text-xl">✕</span>
            </button>
          </div>
        </div>

      {/* Navigation Items */}
      <nav className="p-3 space-y-0.5 pb-20 mb-16">
        {loadingPolicyTypes && navItems.length === 0 ? (
          <div className="text-center text-blue-200 py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
            <p className="text-sm">Loading...</p>
          </div>
        ) : (
          navItems.map((item) => (
          <div
            key={item.id}
            onMouseEnter={() => setHoveredItem(item.id)}
            onMouseLeave={() => setHoveredItem(null)}
            className="relative"
          >
            {/* Main Item */}
            <div className="flex items-center group">
              <Link
                href={item.href}
                onClick={handleLinkClick}
                className={`flex-1 flex items-center gap-2.5 px-3 py-2.5 rounded-lg transition-all duration-300 transform ${
                  isActive(item.href)
                    ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg scale-[1.02]"
                    : hoveredItem === item.id
                    ? "bg-blue-700 text-white shadow-md scale-[1.01] translate-x-0.5"
                    : "text-blue-100 hover:bg-blue-700/50 hover:text-white"
                }`}
              >
                <span className={`text-lg transition-transform duration-300 ${
                  hoveredItem === item.id ? "scale-110" : ""
                }`}>
                  {item.icon}
                </span>
                <span className="font-medium text-[13px] flex-1">{item.label}</span>
                {item.badge && (
                  <span className={`bg-red-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-md transition-all duration-300 ${
                    hoveredItem === item.id ? "scale-105 animate-pulse" : ""
                  }`}>
                    {item.badge}
                  </span>
                )}
              </Link>

              {/* Expand/Collapse Button */}
              {item.submenu && (
                <button
                  onClick={() => toggleExpand(item.id)}
                  className={`px-2 py-2.5 text-blue-100 hover:text-white transition-all duration-300 ${
                    hoveredItem === item.id ? "text-white" : ""
                  }`}
                >
                  <span className={`inline-block transition-transform duration-300 text-xs ${
                    expandedItems.includes(item.id) ? "rotate-180" : ""
                  }`}>
                    ▼
                  </span>
                </button>
              )}
            </div>

            {/* Submenu */}
            {item.submenu && expandedItems.includes(item.id) && (
              <div className="ml-5 mt-0.5 mb-1 space-y-0.5 border-l-2 border-blue-500/50 pl-2.5 animate-slideDown">
                {item.submenu.map((subitem, index) => (
                  <Link
                    key={subitem.id}
                    href={subitem.href}
                    onClick={handleLinkClick}
                    className={`flex items-center gap-2 px-2.5 py-2 rounded-lg text-[12px] transition-all duration-300 transform ${
                      isActive(subitem.href)
                        ? "bg-gradient-to-r from-blue-500 to-blue-400 text-white shadow-md scale-[1.02] translate-x-0.5"
                        : "text-blue-200 hover:bg-blue-700/70 hover:text-white hover:translate-x-0.5 hover:shadow-sm"
                    }`}
                    style={{
                      animationDelay: `${index * 50}ms`,
                    }}
                  >
                    <span className="text-sm transition-transform duration-300 hover:scale-110">
                      {subitem.icon}
                    </span>
                    <span className="font-medium">{subitem.label}</span>
                  </Link>
                ))}
              </div>
            )}
          </div>
          ))
        )}
      </nav>

      {/* Custom CSS for animations */}
      <style jsx>{`
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-slideDown {
          animation: slideDown 0.3s ease-out;
        }
      `}</style>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 w-64 p-2 border-t border-blue-700 bg-gradient-to-r from-blue-900 to-blue-800 backdrop-blur-sm">
          <div className="text-[9px] text-blue-200 text-center space-y-0.5">
            <div className="flex items-center justify-center gap-1 mb-0.5">
              <div className="h-1 w-1 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-300 font-medium text-[10px]">Online</span>
            </div>
            <p className="font-bold text-blue-100 text-[10px]">© 2025 LEGEAI</p>
          </div>
        </div>
      </aside>
    </>
  );
}

