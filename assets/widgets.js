// Each widget: Widgets.name(containerElement). Add new ones here, then use `:::widget name | Title` in a chapter.
const Widgets={
 projection2d(el){
  const cv=document.createElement('canvas'),ro=document.createElement('div');ro.className='ro';el.append(cv,ro);
  const W=cv.width=640,H=cv.height=380,ctx=cv.getContext('2d'),S=50,O=[W/2,H/2];let b=[3,1],x=[1.5,2.5],drag=false;
  const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n),P=v=>[O[0]+v[0]*S,O[1]-v[1]*S];
  const arrow=(f,t,c,dash)=>{const[a,z]=[P(f),P(t)];ctx.strokeStyle=ctx.fillStyle=c;ctx.setLineDash(dash?[6,5]:[]);ctx.lineWidth=2.5;
   ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...z);ctx.stroke();ctx.setLineDash([]);const g=Math.atan2(z[1]-a[1],z[0]-a[0]);
   ctx.beginPath();ctx.moveTo(...z);ctx.lineTo(z[0]-10*Math.cos(g-.4),z[1]-10*Math.sin(g-.4));ctx.lineTo(z[0]-10*Math.cos(g+.4),z[1]-10*Math.sin(g+.4));ctx.fill()};
  function draw(){ctx.clearRect(0,0,W,H);ctx.strokeStyle=css('--line');ctx.lineWidth=1;
   for(let i=0;i<W;i+=S){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,H);ctx.stroke()}for(let j=0;j<H;j+=S){ctx.beginPath();ctx.moveTo(0,j);ctx.lineTo(W,j);ctx.stroke()}
   const dot=(u,v)=>u[0]*v[0]+u[1]*v[1],lam=dot(x,b)/dot(b,b),p=[lam*b[0],lam*b[1]],e=[x[0]-p[0],x[1]-p[1]];
   const k=8/Math.hypot(...b);ctx.strokeStyle=css('--mute');ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(...P([-b[0]*k,-b[1]*k]));ctx.lineTo(...P([b[0]*k,b[1]*k]));ctx.stroke();ctx.setLineDash([]);
   arrow([0,0],b,css('--b'));arrow([0,0],p,css('--p'));arrow(p,x,css('--e'),1);arrow([0,0],x,css('--x'));
   const q=P(x);ctx.beginPath();ctx.arc(q[0],q[1],9,0,7);ctx.fillStyle=css('--x');ctx.globalAlpha=.3;ctx.fill();ctx.globalAlpha=1;
   const f=n=>n.toFixed(2);ro.innerHTML=`x=(${f(x[0])}, ${f(x[1])}) · λ=${f(lam)} · p=(${f(p[0])}, ${f(p[1])})<br>e·b = ${f(dot(e,b))}  ← always 0 (e ⟂ b)`}
  const pos=ev=>{const r=cv.getBoundingClientRect(),t=ev.touches?ev.touches[0]:ev;return[((t.clientX-r.left)*W/r.width-O[0])/S,-((t.clientY-r.top)*H/r.height-O[1])/S]};
  const mv=ev=>{if(drag){x=pos(ev);draw();ev.preventDefault()}};
  cv.onpointerdown=ev=>{drag=true;cv.setPointerCapture(ev.pointerId);mv(ev)};cv.onpointermove=mv;cv.onpointerup=()=>drag=false;
  [['λ > 0',[1.5,2.5]],['λ < 0',[-2,-1]],['λ = 0',[-1,3]],['e = 0',[4.5,1.5]]].forEach(([n,v])=>{const bt=document.createElement('button');bt.textContent=n;bt.onclick=()=>{x=v;draw()};el.append(bt)});
  draw()}
};
