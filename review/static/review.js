// review/static/review.js — robust, emoji-safe, with first-paint self-heal
(function () {
    const textEl =
        document.getElementById('text') ||
        document.querySelector('pre[data-offset], pre[data-base]');
    const listEl = document.getElementById('findingsList');
    const toggleEl = document.getElementById('toggleAll');
    if (!textEl) return;

    // --- Base absolute offset for this scene (index into work.norm_text)
    const BASE = parseInt(textEl.dataset.offset || textEl.dataset.base || '0', 10);

    // --- Get the scene text (prefer rendered <pre>, fallback to JSON blob)
    let ORIGINAL = (textEl.textContent || '').toString();
    if (!ORIGINAL.trim()) {
        const txtTag = document.getElementById('scene-text');
        if (txtTag && txtTag.textContent) {
            try {
                ORIGINAL = JSON.parse(txtTag.textContent) || '';
            } catch {
            }
        }
    }
    if (!ORIGINAL) return;

    // Paint the raw text first
    textEl.textContent = ORIGINAL;

    // --- Build code-point -> code-unit map (emoji-safe indices)
    const cps = Array.from(ORIGINAL);
    const cpLen = cps.length;
    const cpToCu = new Array(cpLen + 1);
    let cu = 0;
    cpToCu[0] = 0;
    for (let i = 0; i < cpLen; i++) {
        cu += cps[i].length;
        cpToCu[i + 1] = cu;
    }

    const escapeHtml = (s) =>
        s.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

    function toSceneCU(absStart, absEnd) {
        let sCP = Math.max(0, Math.min(cpLen, absStart - BASE));
        let eCP = Math.max(0, Math.min(cpLen, absEnd - BASE));
        if (eCP <= sCP) return null;
        return [cpToCu[sCP], cpToCu[eCP]];
    }

    function collectSpans() {
        // Prefer data on cards
        if (listEl) {
            const arr = [...listEl.querySelectorAll('.finding')].map(c => ({
                fid: c.dataset.fid || '',
                start: parseInt(c.dataset.start, 10),
                end: parseInt(c.dataset.end, 10),
                trope: c.dataset.trope || '',
                conf: parseFloat(c.dataset.conf || '0')
            })).filter(x => Number.isFinite(x.start) && Number.isFinite(x.end));
            if (arr.length) return arr;
        }
        // Fallback: JSON blob
        const tag = document.getElementById('spans-json');
        if (tag && tag.textContent) {
            try {
                const arr = JSON.parse(tag.textContent);
                if (Array.isArray(arr)) return arr;
            } catch {
            }
        }
        return [];
    }

    function renderSingle(absStart, absEnd) {
        const rel = toSceneCU(absStart, absEnd);
        if (!rel) return;
        const [s, e] = rel, t = ORIGINAL;
        const head = escapeHtml(t.slice(0, s));
        const mid = escapeHtml(t.slice(s, e));
        const tail = escapeHtml(t.slice(e));
        textEl.innerHTML = head + `<mark class="hl active" id="activeHL">${mid}</mark>` + tail;
        document.getElementById('activeHL')?.scrollIntoView({block: 'center', behavior: 'smooth'});
    }

    function renderAll(spans, activeFid = null) {
        const mapped = spans.map(x => {
            const rel = toSceneCU(x.start, x.end);
            return rel ? {s: rel[0], e: rel[1], fid: x.fid} : null;
        }).filter(Boolean).sort((a, b) => a.s - b.s || b.e - a.e);

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
            textEl.querySelector(`mark[data-fid="${activeFid}"]`)?.scrollIntoView({
                block: 'center',
                behavior: 'smooth'
            });
        }
    }

    const spans = collectSpans();
    let activeCard = null;

    // Self-heal on first paint: if the pane is suspiciously tiny, render all
    requestAnimationFrame(() => {
        const tiny = textEl.clientHeight < 40;   // ~one line or clamped
        if (tiny && spans.length) {
            if (toggleEl) toggleEl.checked = true; // reflect actual state
            renderAll(spans);
        }
    });

    if (listEl) {
        listEl.addEventListener('click', (ev) => {
            const card = ev.target.closest('.finding');
            if (!card) return;

            activeCard?.classList.remove('active');
            card.classList.add('active');
            activeCard = card;

            const start = parseInt(card.dataset.start, 10);
            const end = parseInt(card.dataset.end, 10);

            if (toggleEl?.checked) {
                renderAll(spans, card.dataset.fid);
            } else {
                renderSingle(start, end);
            }

            if (ev.target.classList.contains('jump')) return;
            if (ev.target.classList.contains('accept')) {
                ev.target.textContent = 'Accepted';
                return;
            }
            if (ev.target.classList.contains('reject')) {
                ev.target.textContent = 'Rejected';
                return;
            }
        });
    }

    toggleEl?.addEventListener('change', () => {
        if (toggleEl.checked) {
            renderAll(spans, activeCard?.dataset.fid || null);
        } else {
            textEl.textContent = ORIGINAL;
            if (activeCard) {
                renderSingle(parseInt(activeCard.dataset.start, 10), parseInt(activeCard.dataset.end, 10));
            }
        }
    });
})();
