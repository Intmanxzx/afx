/* Browser credentials stay in a Secure HttpOnly cookie, never browser storage. */
(() => {
  'use strict';
  const host = document.getElementById('session-controls');
  if (!host) return; // Offline viewer has no identity or control surface.
  let profile = null;
  const label = document.createElement('span');
  const button = document.createElement('button');
  button.type = 'button'; button.textContent = '登录';
  host.append(label, button);
  const dialog = document.createElement('dialog');
  dialog.setAttribute('aria-labelledby', 'login-title');
  const form = document.createElement('form');
  const title = document.createElement('h2'); title.id = 'login-title'; title.textContent = 'Token 登录';
  const input = document.createElement('input');
  input.type = 'password'; input.autocomplete = 'off'; input.required = true;
  input.maxLength = 256; input.name = 'token'; input.setAttribute('aria-label', 'Token'); input.placeholder = 'Token';
  const error = document.createElement('p'); error.setAttribute('role', 'alert');
  const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = '登录';
  const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = '取消';
  const actions = document.createElement('div'); actions.className = 'login-actions';
  cancel.className = 'secondary'; actions.append(cancel, submit);
  dialog.className = 'login-dialog';
  form.append(title, input, error, actions); dialog.append(form); document.body.append(dialog);
  function update(value) {
    profile = value;
    label.textContent = value ? value.display_name : '';
    button.textContent = value ? '退出' : '登录';
    document.body.classList.toggle('authenticated', Boolean(value));
    document.dispatchEvent(new CustomEvent('af-session-changed', {detail: value ? {user_id:value.user_id, display_name:value.display_name} : null}));
  }
  async function request(path, options = {}) {
    const method = options.method || 'GET';
    const headers = {...options.headers};
    if (method !== 'GET') {
      headers['Content-Type'] = 'application/json';
      if (profile) headers['X-CSRF-Token'] = profile.csrf;
    }
    const response = await fetch(path, {...options, headers, credentials:'same-origin', cache:'no-store'});
    const result = await response.json();
    if (response.status === 401) update(null);
    if (!response.ok) throw new Error(result.error || '请求失败');
    return result;
  }
  // Future protected controls use this wrapper, not a user id supplied by JS.
  window.afSession = {request, authenticated:() => Boolean(profile)};
  button.addEventListener('click', async () => {
    if (!profile) {error.textContent = ''; dialog.showModal(); input.focus(); return;}
    button.disabled = true;
    try {await request('api/session', {method:'DELETE'}); update(null);}
    catch (e) {label.textContent = e.message;}
    finally {button.disabled = false;}
  });
  cancel.addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {input.value = ''; error.textContent = '';});
  form.addEventListener('submit', async event => {
    event.preventDefault(); submit.disabled = true; cancel.disabled = true; error.textContent = '';
    // Clear the input immediately; only the in-flight request retains the value.
    const body = JSON.stringify({token:input.value.trim()}); input.value = '';
    try {update(await request('api/session', {method:'POST', body})); dialog.close();}
    catch (e) {error.textContent = e.message; input.focus();}
    finally {submit.disabled = false; cancel.disabled = false;}
  });
  async function refresh() {
    try {update(await request('api/session'));}
    catch (_) {update(null);}
  }
  document.addEventListener('visibilitychange', () => {if (!document.hidden) refresh();});
  refresh();
})();
