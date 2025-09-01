// review/static/review.js — robust highlighter + Accept/Reject POST + first-paint self-heal
(function () {
  const textEl =
    document.getElementById('text') ||
    document.querySelector('pre[data-offset], pre[data-base]');
  const listEl   = document.getElementById('findingsList');
  const toggleEl = document.getElementById('toggleAll');
  if (!textEl) return;

  // --- Base offset (absolute char index into work.norm_text)
  const BASE = parseInt(textEl.dataset.offset || textEl.dataset.base || '0', 10);

  // --- Scene text (prefer rendered <pre>; fallback to JSON blob)
  let ORIGINAL = (textEl.textContent || '').toString();
  if (!ORIGINAL.trim()) {
    const txtTag = document.getElementById('scene-text');
    if (txtTag && txtTag.textContent) {
      try { ORIGINAL = JSON.parse(txtTag.textContent) || ''; } catch {}
    }
  }
  if (!ORIGINAL) return;

  // Paint raw text first
  textEl.textContent = ORIGINAL;

  // --- Build code point -> code unit map (emoji-safe)
  const cps   = Array.from(ORIGINAL);
  const cpLen = cps.length;
  const cpToCu = new Array(cpLen + 1);
  let cu = 0; cpToCu[0] = 0;
  for (let i = 0; i < cpLen; i++) { cu += cps[i].length; cpToCu[i + 1] = cu; }

  const escapeHtml = (s) =>
    s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[c]));

  // Absolute [start,end) (code points) -> scene-relative [s,e) (code units)
  function toSceneCU(absStart, absEnd) {
    let sCP = Math.max(0, Math.min(cpLen, absStart - BASE));
    let eCP = Math.max(0, Math.min(cpLen, absEnd   - BASE));
    if (eCP <= sCP) return null;
    return [cpToCu[sCP], cpToCu[eCP]];
  }

  function collectSpans() {
    // Prefer data on cards
    if (listEl) {
      const arr = [...listEl.querySelectorAll('.finding')].map(c => ({
        fid:   c.dataset.fid || '',
        start: parseInt(c.dataset.start, 10),
        end:   parseInt(c.dataset.end,   10),
        trope: c.dataset.trope || '',
        conf:  parseFloat(c.dataset.conf || '0')
      })).filter(x => Number.isFinite(x.start) && Number.isFinite(x.end));
      if (arr.length) return arr;
    }
    // Fallback JSON blob if present
    const tag = document.getElementById('spans-json');
    if (tag && tag.textContent) {
      try { const arr = JSON.parse(tag.textContent); if (Array.isArray(arr)) return arr; } catch {}
    }
    return [];
  }

  function renderSingle(absStart, absEnd) {
    const rel = toSceneCU(absStart, absEnd); if (!rel) return;
    const [s, e] = rel, t = ORIGINAL;
    const head = escapeHtml(t.slice(0, s));
    const mid  = escapeHtml(t.slice(s, e));
    const tail = escapeHtml(t.slice(e));
    textEl.innerHTML = head + `<mark class="hl active" id="activeHL">${mid}</mark>` + tail;
    document.getElementById('activeHL')?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  function renderAll(spans, activeFid = null) {
    const mapped = spans.map(x => {
      const rel = toSceneCU(x.start, x.end);
      return rel ? { s: rel[0], e: rel[1], fid: x.fid } : null;
    }).filter(Boolean).sort((a,b) => a.s - b.s || b.e - a.e);

    let out = [], pos = 0, t = ORIGINAL;
    for (const r of mapped) {
      if (r.s > pos) out.push(escapeHtml(t.slice(pos, r.s)));
      const cls = r.fid === activeFid ? 'hl active' : 'hl';
      out.push(`<mark class="${cls}" data-fid="${r.fid}">${escapeHtml(t.slice(r.s, r.e))}</mark>`);
      pos = r.e;
    }
    out.push(escapeHtml(t.slice(pos)));
    textEl.innerHTML = out.join('');
    if (activeFid) {
      textEl.querySelector(`mark[data-fid="${activeFid}"]`)?.scrollIntoView({ block:'center', behavior:'smooth' });
    }
  }

  const spans = collectSpans();
  let activeCard = null;

  // --- First-paint self-heal: if pane is tiny, show all once
  requestAnimationFrame(() => {
    const tiny = textEl.clientHeight < 40;
    if (tiny && spans.length) {
      if (toggleEl) toggleEl.checked = true;
      renderAll(spans);
    }
  });

  // --- Card clicks (Jump / Accept / Reject)
  if (listEl) {
    listEl.addEventListener('click', async (ev) => {
      const card = ev.target.closest('.finding');
      if (!card) return;

      const isJump    = ev.target.classList.contains('jump');
      const isAccept  = ev.target.classList.contains('accept');
      const isReject  = ev.target.classList.contains('reject');

      // Select/highlight card
      if (isJump || isAccept || isReject) {
        activeCard?.classList.remove('active');
        card.classList.add('active');
        activeCard = card;
      }

      // Jump
      if (isJump) {
        const start = parseInt(card.dataset.start, 10);
        const end   = parseInt(card.dataset.end,   10);
        if (toggleEl?.checked) {
          renderAll(spans, card.dataset.fid);
        } else {
          renderSingle(start, end);
        }
        return;
      }

      // Accept/Reject → POST /api/decision
      if (isAccept || isReject) {
        const fid = card.dataset.fid;
        const decision = isAccept ? 'accept' : 'reject';
        const btn = ev.target;

        const prevLabel = btn.textContent;
        btn.disabled = true;
        btn.textContent = decision === 'accept' ? 'Accepting…' : 'Rejecting…';

        try {
          const res = await fetch('/api/decision', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ finding_id: fid, decision })
          });
          const ok = res.ok && (await res.json()).ok;
          if (ok) {
            btn.textContent = decision === 'accept' ? 'Accepted' : 'Rejected';
            card.classList.remove('accepted','rejected');
            card.classList.add(decision === 'accept' ? 'accepted' : 'rejected');
          } else {
            btn.textContent = 'Error';
            setTimeout(() => (btn.textContent = prevLabel), 1200);
          }
        } catch {
          btn.textContent = 'Error';
          setTimeout(() => (btn.textContent = prevLabel), 1200);
        } finally {
          btn.disabled = false;
        }
        return;
      }
    });
  }

  // --- Toggle show-all
  toggleEl?.addEventListener('change', () => {
    if (toggleEl.checked) {
      renderAll(spans, activeCard?.dataset.fid || null);
    } else {
      textEl.textContent = ORIGINAL;
      if (activeCard) {
        const s = parseInt(activeCard.dataset.start, 10);
        const e = parseInt(activeCard.dataset.end,   10);
        renderSingle(s, e);
      }
    }
  });
})();
