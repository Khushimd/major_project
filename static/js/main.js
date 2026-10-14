(function () {
  const input = document.getElementById("imageInput");
  const preview = document.getElementById("imagePreview");
  const prompt = document.querySelector(".upload-prompt");
  if (input && preview) {
    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (!file) return;
      preview.src = URL.createObjectURL(file);
      preview.classList.remove("hidden");
      if (prompt) prompt.classList.add("hidden");
    });
  }

  document.querySelectorAll("[data-hospital-picker]").forEach((picker) => {
    const field = picker.querySelector(".hospital-picker-input");
    const menu = picker.querySelector(".hospital-picker-menu");
    if (!field || !menu) return;
    const options = [...menu.querySelectorAll("[role=option]")];
    const empty = menu.querySelector(".hospital-picker-empty");
    const filter = () => {
      const q = field.value.toLowerCase().trim();
      let shown = 0;
      options.forEach((opt) => {
        const match = !q || (opt.dataset.search || "").includes(q);
        opt.hidden = !match;
        if (match) shown += 1;
      });
      if (empty) empty.hidden = shown > 0;
      menu.hidden = false;
      field.setAttribute("aria-expanded", "true");
    };
    field.addEventListener("focus", filter);
    field.addEventListener("input", filter);
    menu.addEventListener("mousedown", (event) => {
      const opt = event.target.closest("[role=option]");
      if (!opt) return;
      field.value = opt.dataset.name || "";
      menu.hidden = true;
    });
    document.addEventListener("click", (event) => {
      if (!picker.contains(event.target)) menu.hidden = true;
    });
  });

  const nav = document.querySelector(".site-nav");
  const ids = ["home", "about", "how-it-works", "features", "contact"];
  const sections = ids.map((id) => document.getElementById(id)).filter(Boolean);
  if (nav && sections.length) {
    const links = [...nav.querySelectorAll("a")];
    const setActive = (id) => {
      links.forEach((link) => {
        const hash = (link.getAttribute("href") || "").split("#")[1] || "";
        link.classList.toggle("is-active", hash === id);
      });
    };
    const syncFromScroll = () => {
      const line = (document.querySelector(".site-header")?.offsetHeight || 80) + 48;
      let current = sections[0].id;
      let best = -Infinity;
      sections.forEach((section) => {
        const top = section.getBoundingClientRect().top;
        if (top <= line && top >= best) {
          best = top;
          current = section.id;
        }
      });
      const atBottom =
        window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 80;
      if (atBottom) current = sections[sections.length - 1].id;
      setActive(current);
    };
    links.forEach((link) => {
      link.addEventListener("click", () => {
        const id = (link.getAttribute("href") || "").split("#")[1];
        if (id) setActive(id);
      });
    });
    window.addEventListener("scroll", syncFromScroll, { passive: true });
    window.addEventListener("resize", syncFromScroll);
    syncFromScroll();
  }
})();
