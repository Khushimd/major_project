/**
 * Melanoma CDSS — shared frontend behaviors
 */
(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem("cdss-theme");
  if (stored) {
    root.setAttribute("data-theme", stored);
  }

  const toggle = document.getElementById("themeToggle");
  if (toggle) {
    const syncIcon = () => {
      const dark = root.getAttribute("data-theme") === "dark";
      toggle.innerHTML = dark
        ? '<i class="bi bi-sun"></i>'
        : '<i class="bi bi-moon-stars"></i>';
    };
    syncIcon();
    toggle.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("cdss-theme", next);
      syncIcon();
    });
  }

  // Animated counters on home
  document.querySelectorAll("[data-count]").forEach((el) => {
    const target = Number(el.getAttribute("data-count") || 0);
    const duration = 900;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = Math.round(target * eased).toString();
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  // Upload preview
  const input = document.getElementById("imageInput");
  const preview = document.getElementById("imagePreview");
  const prompt = document.querySelector(".upload-prompt");
  if (input && preview) {
    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      preview.src = url;
      preview.classList.remove("d-none");
      if (prompt) prompt.classList.add("d-none");
    });
  }
})();
