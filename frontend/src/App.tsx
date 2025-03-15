import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import LoginPage from "./pages/LoginPage";
import StudentDashboard from "./pages/StudentDashboard";
import OrganizerDashboard from "./pages/OrganizerDashboard";
import TestSession from "./pages/TestSession";

// Simple auth check
const PrivateRoute = ({ children, userType }: { children: React.ReactNode, userType: string }) => {
  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
  const storedUserType = localStorage.getItem('userType');

  if (!isAuthenticated || (userType && storedUserType !== userType)) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner position="top-right" closeButton />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/assessment-demo" element={<Index />} />

          <Route path="/student-dashboard" element={
            <PrivateRoute userType="student">
              <StudentDashboard />
            </PrivateRoute>
          } />

          <Route path="/organizer-dashboard" element={
            <PrivateRoute userType="organizer">
              <OrganizerDashboard />
            </PrivateRoute>
          } />

          <Route path="/test/:testId" element={
            <PrivateRoute userType="student">
              <TestSession />
            </PrivateRoute>
          } />

          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
