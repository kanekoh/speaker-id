// score-charts.js
// 話者識別スコアの可視化（ライブラリ不使用・SVG 直書き）
window.ScoreCharts = (function () {
  const SVG_NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  function el(tag, attrs = {}, text) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function fmt(score) {
    return score === undefined || score === null ? "–" : Number(score).toFixed(3);
  }

  // ---- ツールチップ（ページに1つだけ） ----
  let tooltip;
  function showTooltip(evt, lines) {
    if (!tooltip) {
      tooltip = el("div", { class: "viz-tooltip", role: "tooltip" });
      document.body.appendChild(tooltip);
    }
    tooltip.replaceChildren(...lines.map((l) => el("div", {}, l)));
    tooltip.style.display = "block";
    const pad = 12;
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    const r = tooltip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }
  function hideTooltip() {
    if (tooltip) tooltip.style.display = "none";
  }
  function attachTooltip(target, lines) {
    target.addEventListener("mousemove", (e) => showTooltip(e, lines));
    target.addEventListener("mouseleave", hideTooltip);
    target.addEventListener("touchstart", (e) => showTooltip(e.touches[0], lines), { passive: true });
  }

  // 右端だけ角丸（4px）の横棒パス。左端（基線側）は直角
  function barPath(x, y, w, h) {
    const r = Math.min(4, w, h / 2);
    if (w <= 0) return "";
    return `M${x},${y} h${w - r} a${r},${r} 0 0 1 ${r},${r} v${h - 2 * r} a${r},${r} 0 0 1 ${-r},${r} h${-(w - r)} Z`;
  }

  // ---- 1回の識別結果：登録者ごとのスコア横棒グラフ ----
  function renderBars(container, scores, threshold) {
    container.replaceChildren();
    if (!scores || scores.length === 0) {
      container.appendChild(el("p", { class: "empty" }, "登録者がいません"));
      return;
    }
    // 表示幅そのままで描画し、文字サイズが画面幅で伸び縮みしないようにする
    const W = Math.max(300, container.clientWidth || 600);
    const labelW = Math.min(130, W * 0.3), valueW = 52, rowH = 30, barH = 18, top = 22;
    const plotW = W - labelW - valueW;
    const H = top + rowH * scores.length + 4;
    const xOf = (v) => labelW + Math.max(0, Math.min(1, v)) * plotW;

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "登録者ごとの類似度スコア" });

    // 目盛り（0, 0.5, 1）
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg.appendChild(svgEl("line", { class: "viz-grid", x1: xOf(t), x2: xOf(t), y1: top - 4, y2: H - 4 }));
      const label = svgEl("text", { x: xOf(t), y: 12, "text-anchor": "middle" });
      label.textContent = t;
      svg.appendChild(label);
    }

    scores.forEach((s, i) => {
      const y = top + i * rowH + (rowH - barH) / 2;
      const name = svgEl("text", { x: labelW - 8, y: y + barH / 2 + 4, "text-anchor": "end" });
      name.textContent = s.kana && s.kana !== s.name ? `${s.kana}` : s.name;
      svg.appendChild(name);

      const w = xOf(s.score) - labelW;
      const bar = svgEl("path", { class: "viz-bar" + (i === 0 ? " is-top" : ""), d: barPath(labelW, y, w, barH) });
      svg.appendChild(bar);

      const value = svgEl("text", { class: "viz-value", x: labelW + plotW + 8, y: y + barH / 2 + 4 });
      value.textContent = fmt(s.score);
      svg.appendChild(value);

      // ホバー領域は行全体
      const hit = svgEl("rect", { class: "viz-hit", x: 0, y: top + i * rowH, width: W, height: rowH });
      attachTooltip(hit, [`${s.kana || s.name}（${s.name}）`, `スコア: ${fmt(s.score)}`,
        s.score >= threshold ? "閾値以上" : "閾値未満"]);
      svg.appendChild(hit);
    });

    // 閾値ライン
    svg.appendChild(svgEl("line", { class: "viz-threshold", x1: xOf(threshold), x2: xOf(threshold), y1: top - 4, y2: H - 4 }));

    container.appendChild(svg);
    container.appendChild(legend());
  }

  function legend(withDots) {
    const box = el("div", { class: "viz-legend" });
    const item = (svgContent, text) => {
      const span = el("span");
      const s = svgEl("svg", { width: 18, height: 12, viewBox: "0 0 18 12", "aria-hidden": "true" });
      svgContent.forEach((c) => s.appendChild(c));
      span.appendChild(s);
      span.appendChild(document.createTextNode(text));
      box.appendChild(span);
    };
    item([svgEl("line", { class: "viz-threshold", x1: 0, x2: 18, y1: 6, y2: 6 })], "閾値");
    if (withDots) {
      item([svgEl("circle", { class: "viz-dot is-hit", cx: 9, cy: 6, r: 4 })], "この人と判定された回");
      item([svgEl("circle", { class: "viz-dot", cx: 9, cy: 6, r: 4 })], "判定されなかった回");
    }
    return box;
  }

  // ---- 人ごとのスコア推移（スモールマルチプル） ----
  function renderMultiples(container, data, maxPoints = 50) {
    container.replaceChildren();
    const entries = data.entries.slice(-maxPoints);
    if (data.people.length === 0 || entries.length === 0) {
      container.appendChild(el("p", { class: "empty" }, "まだ識別の記録がありません"));
      return;
    }
    const grid = el("div", { class: "viz-multiples" });
    const W = 280, H = 120, ml = 30, mr = 8, mt = 8, mb = 8;
    const yMin = -0.2, yMax = 1;
    const n = entries.length;
    const xOf = (i) => ml + (n === 1 ? (W - ml - mr) / 2 : (i / (n - 1)) * (W - ml - mr));
    const yOf = (v) => mt + (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * (H - mt - mb);

    for (const person of data.people) {
      const pts = entries
        .map((e, i) => ({ i, e, score: e.scores[person.name] }))
        .filter((p) => p.score !== undefined);
      const hits = pts.filter((p) => p.e.result === person.name).length;
      const latest = pts.length ? pts[pts.length - 1].score : undefined;
      const max = pts.length ? Math.max(...pts.map((p) => p.score)) : undefined;

      const card = el("div", { class: "viz-card" });
      card.appendChild(el("h3", {}, person.kana && person.kana !== person.name
        ? `${person.kana}（${person.name}）` : person.name));
      card.appendChild(el("p", { class: "viz-stats" },
        `直近 ${fmt(latest)} ／ 最大 ${fmt(max)} ／ 判定 ${hits}回`));

      const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
        "aria-label": `${person.kana || person.name} のスコア推移` });
      for (const t of [0, 0.5, 1]) {
        svg.appendChild(svgEl("line", { class: "viz-grid", x1: ml, x2: W - mr, y1: yOf(t), y2: yOf(t) }));
        const label = svgEl("text", { x: ml - 6, y: yOf(t) + 4, "text-anchor": "end" });
        label.textContent = t;
        svg.appendChild(label);
      }
      svg.appendChild(svgEl("line", { class: "viz-threshold", x1: ml, x2: W - mr,
        y1: yOf(data.threshold), y2: yOf(data.threshold) }));

      // 連続した試行だけを線でつなぐ（登録前の回は空白）
      let d = "";
      pts.forEach((p, k) => {
        const cmd = k > 0 && pts[k - 1].i === p.i - 1 ? "L" : "M";
        d += `${cmd}${xOf(p.i).toFixed(1)},${yOf(p.score).toFixed(1)} `;
      });
      if (d) svg.appendChild(svgEl("path", { class: "viz-line", d }));

      for (const p of pts) {
        const isHit = p.e.result === person.name;
        svg.appendChild(svgEl("circle", { class: "viz-dot" + (isHit ? " is-hit" : ""),
          cx: xOf(p.i), cy: yOf(p.score), r: 4 }));
        const hit = svgEl("circle", { class: "viz-hit", cx: xOf(p.i), cy: yOf(p.score), r: 10 });
        attachTooltip(hit, [
          p.e.time.replace("T", " "),
          `スコア: ${fmt(p.score)}`,
          `判定結果: ${p.e.result ?? "該当なし"}`,
          `送信元: ${p.e.source}`,
        ]);
        svg.appendChild(hit);
      }
      card.appendChild(svg);
      grid.appendChild(card);
    }
    container.appendChild(grid);
    container.appendChild(legend(true));
  }

  // ---- ログ表（新しい順） ----
  function renderTable(container, data) {
    container.replaceChildren();
    if (data.entries.length === 0) {
      container.appendChild(el("p", { class: "empty" }, "まだ識別の記録がありません"));
      return;
    }
    const wrap = el("div", { class: "table-wrap" });
    const table = el("table", { class: "data-table" });
    const head = el("tr");
    ["時刻", "送信元", "判定", "声の長さ", "最高スコア", ...data.people.map((p) => p.kana || p.name)]
      .forEach((h) => head.appendChild(el("th", {}, h)));
    table.appendChild(el("thead")).appendChild(head);
    const body = el("tbody");
    for (const e of [...data.entries].reverse()) {
      const tr = el("tr");
      tr.appendChild(el("td", {}, e.time.replace("T", " ")));
      tr.appendChild(el("td", {}, e.source));
      tr.appendChild(e.result
        ? el("td", { class: "viz-status-ok" }, `✓ ${e.result}`)
        : e.reason === "too_short"
          ? el("td", { class: "viz-status-ng" }, "✗ 短すぎ")
          : el("td", { class: "viz-status-ng" }, `✗ 該当なし${e.candidate ? `（${e.candidate}）` : ""}`));
      tr.appendChild(el("td", { class: "num" }, e.speech_sec == null ? "–" : `${e.speech_sec.toFixed(1)}秒`));
      tr.appendChild(el("td", { class: "num" }, fmt(e.score)));
      for (const p of data.people) tr.appendChild(el("td", { class: "num" }, fmt(e.scores[p.name])));
      body.appendChild(tr);
    }
    table.appendChild(body);
    wrap.appendChild(table);
    container.appendChild(wrap);
  }

  return { renderBars, renderMultiples, renderTable };
})();
