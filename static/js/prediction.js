/**
 * Prediction page — confidence gauge + probability chart
 */
(function () {
  const data = window.CDSS_RESULT;
  if (!data || typeof Chart === "undefined") return;

  const conf = Math.max(0, Math.min(1, Number(data.confidence) || 0));
  const malignant = Number(data.malignant) || 0;
  const benign = Number(data.benign) || 0;
  const isMalignant = String(data.prediction).toLowerCase() === "malignant";
  const color = isMalignant ? "#b42318" : "#0f7a4c";

  const gaugeCtx = document.getElementById("confidenceGauge");
  if (gaugeCtx) {
    new Chart(gaugeCtx, {
      type: "doughnut",
      data: {
        datasets: [
          {
            data: [conf, 1 - conf],
            backgroundColor: [color, "rgba(120,140,150,0.18)"],
            borderWidth: 0,
            cutout: "78%",
          },
        ],
      },
      options: {
        responsive: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { animateRotate: true, duration: 1100 },
      },
    });
  }

  const probCtx = document.getElementById("probChart");
  if (probCtx) {
    new Chart(probCtx, {
      type: "bar",
      data: {
        labels: ["Benign", "Malignant"],
        datasets: [
          {
            label: "Probability",
            data: [benign, malignant],
            backgroundColor: ["#0f7a4c", "#b42318"],
            borderRadius: 8,
            maxBarThickness: 48,
          },
        ],
      },
      options: {
        indexAxis: "y",
        scales: {
          x: {
            min: 0,
            max: 1,
            ticks: {
              callback: (v) => `${Math.round(v * 100)}%`,
            },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${(ctx.raw * 100).toFixed(1)}%`,
            },
          },
        },
      },
    });
  }
})();
