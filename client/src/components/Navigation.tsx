import { Link, useLocation } from "wouter";
import { Activity, LayoutDashboard, ScanFace, Home } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

export default function Navigation() {
  const [location] = useLocation();

  const navItems = [
    { href: "/", label: "HOME", icon: Home },
    { href: "/scan", label: "SCAN", icon: ScanFace },
    { href: "/dashboard", label: "DASHBOARD", icon: LayoutDashboard },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 h-16 z-50 glass border-b border-black/5">
      <div className="max-w-7xl mx-auto px-4 h-full flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
            <Activity className="w-5 h-5" />
          </div>
          <span className="font-display font-semibold text-lg tracking-tight">
            AI Silent Disease Predictor
          </span>
        </div>

        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = location === item.href;
            return (
              <Link key={item.href} href={item.href} className="relative px-4 py-2 text-sm font-medium transition-colors hover:text-primary rounded-full group outline-none">
                <div className="flex items-center gap-2 relative z-10">
                  <item.icon className={cn("w-4 h-4", isActive ? "text-primary" : "text-muted-foreground group-hover:text-primary")} />
                  <span className={isActive ? "text-primary" : "text-muted-foreground group-hover:text-primary"}>
                    {item.label}
                  </span>
                </div>
                {isActive && (
                  <motion.div
                    layoutId="nav-active"
                    className="absolute inset-0 bg-primary/10 rounded-full"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}