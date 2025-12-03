import { baseUrl } from "./baseUrl";
import { SWRConfig } from "swr";
import { WsProvider } from "./ws";
import axios from "axios";
import { ReactNode } from "react";

axios.defaults.baseURL = `${baseUrl}api/`;

// Module-level flag to prevent multiple simultaneous redirects
// (eg, when multiple SWR queries fail with 401 at once)
// Fixes iOS PWA redirect loop issues
let isRedirectingToLogin = false;

type ApiProviderType = {
  children?: ReactNode;
  options?: Record<string, unknown>;
};

export function ApiProvider({ children, options }: ApiProviderType) {
  axios.defaults.headers.common = {
    "X-CSRF-TOKEN": 1,
    "X-CACHE-BYPASS": 1,
  };

  return (
    <SWRConfig
      value={{
        fetcher: (key) => {
          const [path, params] = Array.isArray(key) ? key : [key, undefined];
          return axios.get(path, { params }).then((res) => res.data);
        },
        onError: (error, _key) => {
          if (
            error.response &&
            [401, 302, 307].includes(error.response.status)
          ) {
            // Redirect to login if not already there and not already redirecting
            if (
              !window.location.pathname.endsWith("/login") &&
              !isRedirectingToLogin
            ) {
              isRedirectingToLogin = true;
              window.location.href = "/login";
            }
          }
        },
        ...options,
      }}
    >
      <WsWithConfig>{children}</WsWithConfig>
    </SWRConfig>
  );
}

type WsWithConfigType = {
  children: ReactNode;
};

function WsWithConfig({ children }: WsWithConfigType) {
  return <WsProvider>{children}</WsProvider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useApiHost() {
  return baseUrl;
}
