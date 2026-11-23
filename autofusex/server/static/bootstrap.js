"use strict";
// Both hosts serve this shell and the same packaged assets. The host declares capabilities.
const response = await fetch("api/capabilities", {cache: "no-store"});
if (!response.ok) throw new Error("Unable to read viewer capabilities");
const capabilities = await response.json();
if (capabilities.mode === "local") {
  const page = await fetch("local.html", {cache: "no-store"});
  if (!page.ok) throw new Error("Unable to load local results view");
  document.querySelector('link[rel="stylesheet"]').href = "local.css";
  // Trusted packaged markup only; result values are rendered by the local view as text.
  document.body.innerHTML = await page.text();
  document.title = "AFX 本地结果";
  for (const name of ["charts.js", "local.js", "runtime.js"]) {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = name; script.onload = resolve; script.onerror = reject;
      document.head.append(script);
    });
  }
} else {
  await import("./app.js");
}
