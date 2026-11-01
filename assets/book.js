const Book={
 async list(){return (await fetch('chapters.json')).json()},
 math(md){ // protect TeX from the markdown parser
  const s=[];md=md.replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g,(m,d,i)=>{s.push([!!d,d||i]);return '@@M'+(s.length-1)+'@@'});
  return [md,s]},
 unmath(html,s){return html.replace(/@@M(\d+)@@/g,(m,i)=>{const [d,t]=s[i];
  try{return katex.renderToString(t,{displayMode:d,throwOnError:false})}catch(e){return t}}).replace(/<p>(<span class="katex-display">[\s\S]*?<\/span>)<\/p>/g,'<div class="math">$1</div>')},
 figs(md){ // :::figure ID | Title | description   and   :::widget name
  return md.replace(/^:::figure (.+)$/gm,(m,l)=>{const [id,t,d]=l.split('|').map(x=>x.trim());
   return `<div class="fig todo"><span class="tag">${id} · PLACEHOLDER</span><b> ${t}</b><br>${d}</div>`})
  .replace(/^:::widget (\w+)(?: \| (.+))?$/gm,(m,n,t)=>`<div class="fig" data-widget="${n}"><span class="tag">INTERACTIVE</span> <b>${t||''}</b></div>`)},
 async home(){const c=await Book.list();let part='',h='';
  c.forEach(x=>{if(x.part!==part){part=x.part;h+=`<h3>${part}</h3>`}
   h+=`<a href="chapter.html?c=${x.id}"><b>${x.title}</b><span class="badge">${x.status}</span><br><small>${x.summary}</small></a>`});
  toc.innerHTML=h;
  let g='graph LR\n';c.forEach(x=>{g+=` ${x.id.slice(0,2).replace(/^/,'c')}["${+x.id.slice(0,2)}. ${x.title.split(/[&,]/)[0].trim()}"]\n`;
   x.connects.forEach(y=>g+=` c${x.id.slice(0,2)} --> c${y.slice(0,2)}\n`);g+=` click c${x.id.slice(0,2)} "chapter.html?c=${x.id}"\n`});
  map.textContent=g;mermaid.initialize({startOnLoad:false,securityLevel:'loose'});mermaid.run({nodes:[map]})},
 async chapter(){const id=new URLSearchParams(location.search).get('c'),c=await Book.list(),i=c.findIndex(x=>x.id===id),ch=c[i];
  if(!ch){main.textContent='Unknown chapter.';return}document.title=ch.title;
  let body;const r=await fetch(`chapters/${id}.md`);
  if(r.ok){const [md,s]=Book.math(Book.figs(await r.text()));body=Book.unmath(marked.parse(md),s)}
  else{body=`<h1>${ch.title}</h1><div class="callout">Draft outline — full text coming soon.<br>${ch.summary}</div>
   <h2>Prerequisites</h2><ul>${ch.prereq.map(p=>`<li>${p}</li>`).join('')}</ul>
   <h2>Diagrams to build</h2>${ch.figs.map(f=>{const [a,b,d]=f.split('|').map(x=>x.trim());return `<div class="fig todo"><span class="tag">${a} · PLACEHOLDER</span><b> ${b}</b><br>${d}</div>`}).join('')}`}
  const nx=c.slice(i+1).find(()=>1),pv=c[i-1],lk=x=>`chapter.html?c=${x.id}`;
  const rel=ch.connects.map(k=>c.find(x=>x.id===k)).filter(Boolean).map(x=>`<a href="${lk(x)}">${x.title}</a>`).join(' · ');
  main.innerHTML=body+(rel?`<div class="callout"><b>Leads to:</b> ${rel}</div>`:'')+`<div class="nav"><span>${pv?`<a href="${lk(pv)}">← ${pv.title}</a>`:''}</span><span>${nx?`<a href="${lk(nx)}">${nx.title} →</a>`:''}</span></div>`;
  document.querySelectorAll('[data-widget]').forEach(el=>Widgets[el.dataset.widget]&&Widgets[el.dataset.widget](el))}
};
