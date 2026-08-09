/**
 * Model comparison charts
 */
(function () {
  const metrics = window.CDSS_METRICS;
  if (!metrics || typeof Chart === "undefined") return;

  const names = Object.keys(metrics);
  const labels = names.map((n) => n.charAt(0).toUpperCase() + n.slice(1));

  const compareCtx = document.getElementById("compareChart");
  if (compareCtx) {
    new Chart(compareCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Accuracy",
            data: names.map((n) => metrics[n].accuracy),
            backgroundColor: "#0b6e7a",
          },
          {
            label: "Precision",
            data: names.map((n) => metrics[n].precision),
            backgroundColor: "#1f8f9b",
          },
          {
            label: "Recall",
            data: names.map((n) => metrics[n].recall),
            backgroundColor: "#c45c26",
          },
          {
            label: "F1",
            data: names.map((n) => metrics[n].f1),
            backgroundColor: "#e08955",
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: { min: 0, max: 1, ticks: { callback: (v) => `${Math.round(v * 100)}%` } },
        },
      },
    });
  }

  const aucCtx = document.getElementById("aucChart");
  if (aucCtx) {
    new Chart(aucCtx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [
          {
            data: names.map((n) => metrics[n].auc),
            backgroundColor: ["#0b6e7a", "#c45c26", "#1f8f9b"],
          },
        ],
      },
      options: {
        plugins: {
          title: { display: true, text: "ROC-AUC by model" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.label}: ${Number(ctx.raw).toFixed(4)}`,
            },
          },
        },
      },
    });
  }
})();
