import { useContext, useEffect } from "react";
import { Navigate, Outlet } from "react-router-dom";
import { AuthContext } from "@/context/auth-context";
import ActivityIndicator from "../indicators/activity-indicator";
import {
  isRedirectingToLogin,
  setRedirectingToLogin,
} from "@/api/auth-redirect";

export default function ProtectedRoute({
  requiredRoles,
}: {
  requiredRoles: ("admin" | "viewer")[];
}) {
  const { auth } = useContext(AuthContext);

  // Redirect to login page (separate HTML page) when not authenticated
  useEffect(() => {
    if (
      !auth.isLoading &&
      auth.isAuthenticated &&
      !auth.user &&
      !isRedirectingToLogin()
    ) {
      setRedirectingToLogin(true);
      window.location.href = "/login";
    }
  }, [auth.isLoading, auth.isAuthenticated, auth.user]);

  if (auth.isLoading) {
    return (
      <ActivityIndicator className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2" />
    );
  }

  // Unauthenticated mode
  if (!auth.isAuthenticated) {
    return <Outlet />;
  }

  // Authenticated mode (8971): require login
  if (!auth.user) {
    // Redirect happening in useEffect, show loading in the meantime
    return (
      <ActivityIndicator className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2" />
    );
  }

  // If role is null (shouldn’t happen if isAuthenticated, but type safety), fallback
  // though isAuthenticated should catch this
  if (auth.user.role === null) {
    return <Outlet />;
  }

  if (!requiredRoles.includes(auth.user.role)) {
    return <Navigate to="/unauthorized" replace />;
  }

  return <Outlet />;
}
