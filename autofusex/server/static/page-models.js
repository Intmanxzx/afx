"use strict";
import { el, api, rkey, time } from "./core.js";
import { PAGES, render } from "./router.js";

PAGES.models = async root => {
  const data = await api('/api/source-jobs');
  root.append(el('section', {class:'panel'}, el('h2', {}, '导入用例与整网任务'),
    el('p', {}, '通过 afx case import 登记用例，使用 afx case run --submit 提交。结果按用例版本记录，独立于小图统计。')));
  for (const job of data.items) {
    const result = job.result;
    const retry = el('button', {onclick: async () => {
      retry.disabled = true;
      try { await api('/api/source-jobs', {method:'POST', body:{request_key:rkey(), spec:job.spec}}); await render(); }
      finally { retry.disabled = false; }
    }}, '复测此版本');
    const panel = el('section', {class:'panel'}, el('h3', {}, job.case_key),
      el('p', {}, time(job.created_at) + ' · ' + job.state + (result ? ' · ' + result.status : '')),
      el('p', {class:'mono'}, '任务 ' + job.job_id), retry);
    if (result) {
      panel.append(el('pre', {}, JSON.stringify(result.summary || {error:result.error,message:result.message}, null, 2)));
      for (const artifact of result.artifacts || []) {
        if (artifact.size > 1048576 || !/\.(html|json|txt|csv|svg)$/.test(artifact.path)) continue;
        panel.append(el('button', {onclick: async () => {
          const blob = await api('/api/source-jobs/' + encodeURIComponent(job.job_id) + '/artifact?path=' + encodeURIComponent(artifact.path), {raw:true});
          const url=URL.createObjectURL(blob), link=document.createElement('a');
          link.href=url; link.download=artifact.path.split('/').pop(); link.click();
          setTimeout(() => URL.revokeObjectURL(url),1000);
        }}, '下载 ' + artifact.path));
      }
      if (result.artifacts?.length) panel.append(el('details', {}, el('summary', {}, '产物清单'),
        el('pre', {}, JSON.stringify(result.artifacts, null, 2))));
    }
    root.append(panel);
  }
  if (!data.items.length) root.append(el('div', {class:'empty'}, '暂无导入或整网任务'));
};
