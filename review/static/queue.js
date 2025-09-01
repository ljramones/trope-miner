// review/static/queue.js — micro-interactions for the queue
(function () {
  const metaTag = document.getElementById('queue-meta');
  if (!metaTag) return;
  const META = JSON.parse(metaTag.textContent || '{}');
  const filters = META.filters || {};
  const fid = META.current_fid;

  const listEl = document.getElementById('findingsList');
  const card = listEl ? listEl.querySelector('.finding') : null;

  function nextURL(extra) {
    const p = new URLSearchParams();
    if (filters.work_id)  p.set('work_id',  filters.work_id);
    if (filters.trope_id) p.set('trope_id', filters.trope_id);
    if (filters.order)    p.set('order',    filters.order);
    if (filters.min_conf != null) p.set('min_conf', filters.min_conf);
    if (filters.max_conf != null) p.set('max_conf', filters.max_conf);
    if (extra && extra.after) p.set('after', extra.after);
    return '/queue?' + p.toString();
  }

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
    return r.ok ? r.json() : { ok:false };
  }

  async function doDecision(decision) {
    const res = await postJSON('/api/decision', { finding_id: fid, decision });
    if (res && res.ok) location.href = nextURL({ after: fid });
  }

  async function doEdit() {
    const start = parseInt(card?.dataset.start || '0', 10);
    const end   = parseInt(card?.dataset.end   || '0', 10);
    const v = prompt('Enter new absolute start,end (inclusive-exclusive)', `${start},${end}`);
    if (!v) return;
    const parts = v.split(',').map(s => parseInt(s.trim(), 10));
    if (parts.length !== 2 || isNaN(parts[0]) || isNaN(parts[1])) return alert('Bad input');
    const res = await postJSON('/api/edit_span', { finding_id: fid, start: parts[0], end: parts[1] });
    if (res && res.ok) {
      // chain into Accept to mark this as good
      const confirmAccept = confirm('Span updated. Accept this finding now?');
      if (confirmAccept) await doDecision('accept'); else location.href = nextURL({ after: fid });
    }
  }

  listEl?.addEventListener('click', (e) => {
    const t = e.target;
    if (t.classList.contains('accept')) return void doDecision('accept');
    if (t.classList.contains('reject')) return void doDecision('reject');
    if (t.classList.contains('edit'))   return void doEdit();
    if (t.classList.contains('next'))   return void (location.href = nextURL({ after: fid }));
    if (t.classList.contains('jump'))   return; // handled by review.js highlighter
  });

  document.addEventListener('keydown', (e) => {
    // avoid conflicting with inputs
    if (/input|textarea|select/i.test(e.target.tagName)) return;
    if (e.key === 'a' || e.key === 'A') { e.preventDefault(); doDecision('accept'); }
    if (e.key === 'r' || e.key === 'R') { e.preventDefault(); doDecision('reject'); }
    if (e.key === 'e' || e.key === 'E') { e.preventDefault(); doEdit(); }
    if (e.key === 'n' || e.key === 'N' || e.key === ' ') { e.preventDefault(); location.href = nextURL({ after: fid }); }
  });

  // On load, ensure "show all" is checked (it is, in the template)
  document.getElementById('toggleAll')?.dispatchEvent(new Event('change'));
})();
