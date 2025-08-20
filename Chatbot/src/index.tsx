import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { DesktopSizeX } from "./screens/DesktopSizeX";

createRoot(document.getElementById("app") as HTMLElement).render(
  <StrictMode>
    <DesktopSizeX />
  </StrictMode>,
);
