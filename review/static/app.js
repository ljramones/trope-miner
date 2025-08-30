
// put this at the very top of review/static/app.js
document.addEventListener('DOMContentLoaded', () => {
  document.documentElement.classList.add('ready','hydrated','loaded');
  document.body.classList.add('ready','hydrated','loaded');
  document.querySelector('section.content')?.classList.add('ready','hydrated','loaded');
  document.querySelector('.sceneWrap')?.classList.add('ready','hydrated','loaded');
});

function colorFromString(s) {
    // deterministic HSL; stable per id
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return `hsl(${h % 360} 80% 85%)`;
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, m => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'}[m]));
}

function renderHighlights(textEl, text, spans, offset) {
    // sort by start; clip to scene; handle simple overlaps
    spans = spans
        .map(sp => ({
            ...sp,
            r0: Math.max(0, sp.start - offset),
            r1: Math.max(0, sp.end - offset)
        }))
        .filter(sp => sp.r1 > sp.r0)
        .sort((a, b) => a.r0 - b.r0 || a.r1 - b.r1);

    let html = [];
    let cur = 0;
    for (const sp of spans) {
        const s = Math.max(cur, sp.r0);
        const e = Math.max(s, sp.r1);
        if (s > cur) html.push(escapeHtml(text.slice(cur, s)));
        if (e > s) {
            const frag = text.slice(s, e);
            const tip = `${sp.trope} (${(sp.confidence || 0).toFixed(2)}) [${sp.start}–${sp.end}]`;
            html.push(`<mark id="m_${sp.id}" data-fid="${sp.id}" title="${escapeHtml(tip)}" style="background:${colorFromString(sp.id)}">${escapeHtml(frag)}</mark>`);
        }
        cur = Math.max(cur, e);
    }
    if (cur < text.length) html.push(escapeHtml(text.slice(cur)));
    textEl.innerHTML = html.join("");
}

function jumpTo(fid) {
    const el = document.getElementById('m_' + fid);
    if (!el) return;
    el.scrollIntoView({behavior: 'smooth', block: 'center'});
    el.classList.add('pulse');
    setTimeout(() => el.classList.remove('pulse'), 800);
}

async function decide(fid, decision) {
    await fetch('/api/decision', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({finding_id: fid, decision})
    });
    location.reload();
}

async function editSpan(fid) {
    const s = parseInt(document.getElementById('s_' + fid).value, 10);
    const e = parseInt(document.getElementById('e_' + fid).value, 10);
    const tsel = document.getElementById('t_' + fid);
    const trope_id = tsel ? tsel.value : null;
    await fetch('/api/edit_span', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({finding_id: fid, start: s, end: e, trope_id})
    });
    location.reload();
}

async function addFinding(work_id, scene_id) {
    const trope_id = document.getElementById('new_trope').value;
    const start = parseInt(document.getElementById('new_start').value, 10);
    const end = parseInt(document.getElementById('new_end').value, 10);
    const confidence = parseFloat(document.getElementById('new_conf').value);
    const r = await fetch('/api/new_finding', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({work_id, scene_id, trope_id, start, end, confidence})
    });
    const resp = await r.json();
    if (resp.ok) location.reload();
}

function selectionToOffsets(container) {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return null;
    const rng = sel.getRangeAt(0);
    // ignore if selection is outside container
    if (!container.contains(rng.startContainer) || !container.contains(rng.endContainer)) return null;
    const preStart = document.createRange();
    preStart.setStart(container, 0);
    preStart.setEnd(rng.startContainer, rng.startOffset);
    const preEnd = document.createRange();
    preEnd.setStart(container, 0);
    preEnd.setEnd(rng.endContainer, rng.endOffset);
    return [preStart.toString().length, preEnd.toString().length];
}

window.addEventListener('DOMContentLoaded', () => {
    const textEl = document.getElementById('text');
    const offset = parseInt(textEl.dataset.offset, 10);
    const sceneText = window.__SCENE_TEXT__ || '';
    const spans = JSON.parse(document.getElementById('spans-json').textContent || '[]');
    renderHighlights(textEl, sceneText, spans, offset);

    // Fill "Add New" start/end from selection in the text
    textEl.addEventListener('mouseup', () => {
        const sel = selectionToOffsets(textEl);
        if (!sel) return;
        const [sRel, eRel] = sel;
        document.getElementById('new_start').value = offset + sRel;
        document.getElementById('new_end').value = offset + eRel;
    });

    // Clicking a finding card jumps to mark
    document.querySelectorAll('.finding').forEach(card => {
        card.addEventListener('click', (ev) => {
            if (ev.target.closest('.actions,.edit')) return; // don't hijack buttons
            jumpTo(card.dataset.fid);
        });
    });
});
