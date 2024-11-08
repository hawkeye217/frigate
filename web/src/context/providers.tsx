import { ReactNode } from "react";
import { ThemeProvider } from "@/context/theme-provider";
import { RecoilRoot } from "recoil";
import { ApiProvider } from "@/api";
import { IconContext } from "react-icons";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StatusBarMessagesProvider } from "@/context/statusbar-provider";
import { NuqsAdapter } from "nuqs/adapters/react";

type TProvidersProps = {
  children: ReactNode;
};

function providers({ children }: TProvidersProps) {
  return (
    <RecoilRoot>
      <ApiProvider>
        <NuqsAdapter>
          <ThemeProvider defaultTheme="system" storageKey="frigate-ui-theme">
            <TooltipProvider>
              <IconContext.Provider value={{ size: "20" }}>
                <StatusBarMessagesProvider>
                  {children}
                </StatusBarMessagesProvider>
              </IconContext.Provider>
            </TooltipProvider>
          </ThemeProvider>
        </NuqsAdapter>
      </ApiProvider>
    </RecoilRoot>
  );
}

export default providers;
