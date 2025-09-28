async function getJSON(url, options = {}) {
  const res = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
  const text = await res.text();
  try { return JSON.parse(text); } catch { return { raw: text }; }
}

document.getElementById('btnTime').addEventListener('click', async () => {
  const data = await getJSON('/api/time');
  document.getElementById('outTime').textContent = JSON.stringify(data, null, 2);
});

document.getElementById('btnEcho').addEventListener('click', async () => {
  let body;
  try {
    body = JSON.parse(document.getElementById('echoInput').value || '{}');
  } catch (e) {
    body = { invalidJSON: true, text: document.getElementById('echoInput').value };
  }
  const data = await getJSON('/api/echo', { method: 'POST', body: JSON.stringify(body) });
  document.getElementById('outEcho').textContent = JSON.stringify(data, null, 2);
});

document.getElementById('btnHealth').addEventListener('click', async () => {
  const data = await getJSON('/health');
  document.getElementById('outHealth').textContent = JSON.stringify(data, null, 2);
});
