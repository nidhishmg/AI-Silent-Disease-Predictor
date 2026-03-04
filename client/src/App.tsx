import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ScanProvider } from "@/context/ScanContext";
import NotFound from "@/pages/not-found";

import Navigation from "@/components/Navigation";
import Home from "@/pages/Home";
import Scan from "@/pages/Scan";
import Dashboard from "@/pages/Dashboard";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/scan" component={Scan} />
      <Route path="/dashboard" component={Dashboard} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <ScanProvider>
          <div className="min-h-screen flex flex-col relative selection:bg-primary/20">
            <Navigation />
            <main className="flex-1 pt-16">
              <Router />
            </main>
          </div>
          <Toaster />
        </ScanProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;