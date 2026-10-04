'use strict';
const $=id=>document.getElementById(id);
const fields=['level','card_rank','awake_count'];
const fieldNames={level:'等级',card_rank:'卡阶',awake_count:'特训'};
let files=[],result=null,catalog=[],imageURLs=[];
const text=(tag,value,cls)=>{const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e};
function cardStatus(card){
  const missing=fields.filter(k=>k!=='awake_count'||card.kind==='member').some(k=>card[k].value===null);
  const conflicting=Object.keys(card.conflicts||{}).length>0;
  const manual=fields.some(k=>card[k].source==='manual');
  return text('span',conflicting?'存在冲突':missing?(manual?'待补充 · 已编辑':'待补充'):manual?'已校对':'已读取','badge'+(missing||conflicting?' warn':''));
}
fetch('/api/catalog').then(r=>r.json()).then(x=>catalog=x).catch(()=>{$('status').textContent='图库读取失败，请刷新页面。'});

function setFiles(list){
  files=[...list].filter(f=>f.type.startsWith('image/'));
  $('files').textContent=files.map(f=>f.name).join(' · ');
  $('scan').disabled=!files.length||files.length>30;
  if(files.length>30)$('status').textContent='每批最多 30 张，请分批选择。';
}
$('upload').onchange=e=>setFiles(e.target.files);
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('over')};
$('drop').ondragleave=()=>$('drop').classList.remove('over');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('over');setFiles(e.dataTransfer.files)};

function emptyRows(message){$('rows').replaceChildren();const tr=document.createElement('tr'),td=text('td',message,'empty');td.colSpan=6;tr.append(td);$('rows').append(tr)}
function draw(){
  const box=result.box;let review=0;$('rows').replaceChildren();
  for(const card of box.cards){
    const missing=fields.filter(k=>k!=='awake_count'||card.kind==='member').some(k=>card[k].value===null);
    const conflicts=Object.keys(card.conflicts||{});review+=missing||conflicts.length>0;
    const tr=document.createElement('tr'),td=document.createElement('td'),wrap=document.createElement('div');wrap.className='identity';
    const info=catalog.find(c=>c.kind===card.kind&&c.id===card.id),button=document.createElement('button'),img=document.createElement('img');
    button.className='inspect';button.title='查看截图证据';button.setAttribute('aria-label','查看 '+card.name+' 的截图证据');
    if(info)img.src='/art/'+info.file.split('/').at(-1);img.alt='';button.append(img);button.onclick=()=>preview(card);wrap.append(button);
    const name=text('div',card.name);name.append(text('small',(card.kind==='member'?'成员':'Snap')+' #'+card.id));
    if(info?.title)name.append(text('small',info.title));wrap.append(name);td.append(wrap);tr.append(td);
    for(const key of fields){
      const cell=document.createElement('td');
      if(key==='awake_count'&&card.kind==='snap'){cell.append(text('span','—'));tr.append(cell);continue}
      const input=document.createElement('input');input.type='number';input.min=1;
      input.max=key==='level'?(info?.max_level||100):5;input.step=1;
      input.value=card[key].value??'';input.placeholder='未知';input.setAttribute('aria-label',card.name+' '+fieldNames[key]);
      if(card[key].value===null)input.className='unknown';
      input.oninput=()=>{
        $('export').disabled=!!$('rows').querySelector('input:invalid');
        if(!input.checkValidity())return;
        const value=input.value===''?null:Number(input.value);
        if(value===card[key].value&&!card.conflicts[key])return;
        result.box.corrections??=[];
        result.box.corrections.push({kind:card.kind,id:card.id,field:key,previous:structuredClone(card[key]),
          previous_conflicts:structuredClone(card.conflicts[key]||null),value});
        card[key]={value,confidence:value===null?0:1,source:'manual'};
        delete card.conflicts[key];
        input.className=value===null?'unknown':'';
        cell.querySelector('.conflict')?.remove();
        status.replaceChildren(cardStatus(card));
        $('review').textContent=box.cards.filter(c=>Object.keys(c.conflicts||{}).length||fields.filter(k=>k!=='awake_count'||c.kind==='member').some(k=>c[k].value===null)).length;
      };
      input.onchange=()=>{if(!input.checkValidity())input.reportValidity()};
      cell.append(input);
      if(card.conflicts[key])cell.append(text('div','冲突：'+Object.keys(card.conflicts[key]).join(' / '),'conflict'));
      tr.append(cell);
    }
    const status=document.createElement('td');status.append(cardStatus(card));
    tr.append(status,text('td',card.sources.join(' / '),'source'));$('rows').append(tr);
  }
  $('members').textContent=box.cards.filter(c=>c.kind==='member').length;
  $('snaps').textContent=box.cards.filter(c=>c.kind==='snap').length;
  $('duplicates').textContent=box.duplicate_observations;$('review').textContent=review;$('export').disabled=false;
  if(!box.cards.length)emptyRows('没有识别到已知卡片，请使用清晰的成员或 Snap 列表截图。');
}

async function preview(card){
  $('preview-title').textContent=card.name+' · 截图证据';$('preview-body').replaceChildren();$('preview').showModal();
  for(let i=0;i<result.scans.length;i++){
    const scan=result.scans[i],observation=scan.cards.find(c=>c.kind===card.kind&&c.id===card.id);
    if(!observation||!imageURLs[i])continue;
    const figure=document.createElement('figure');figure.append(text('figcaption',scan.source+' · '+fields.map(k=>fieldNames[k]+': '+(observation[k].value??'未读取')).join(' / ')));
    const image=new Image();image.src=imageURLs[i];await image.decode();
    const [x,y,w,h]=observation.bbox,pad=Math.max(12,w*.15),left=Math.max(0,x-pad),top=Math.max(0,y-pad);
    const right=Math.min(image.width,x+w+pad),bottom=Math.min(image.height,y+h+pad);
    const canvas=document.createElement('canvas');canvas.width=Math.round((right-left)*2);canvas.height=Math.round((bottom-top)*2);
    const ctx=canvas.getContext('2d');ctx.drawImage(image,left,top,right-left,bottom-top,0,0,canvas.width,canvas.height);
    ctx.strokeStyle='#18a875';ctx.lineWidth=2;ctx.strokeRect((x-left)*2,(y-top)*2,w*2,h*2);
    figure.append(canvas);$('preview-body').append(figure);
  }
}
$('close-preview').onclick=()=>$('preview').close();
const dataURL=f=>new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve({name:f.name,data:r.result});r.onerror=reject;r.readAsDataURL(f)});
$('scan').onclick=async()=>{
  const selected=[...files];$('scan').disabled=true;$('reset').disabled=true;$('status').textContent='正在识别 '+selected.length+' 张截图…';
  try{
    const images=await Promise.all(selected.map(dataURL)),response=await fetch('/api/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({schema:'ournotes-boxlens.scan-request/1',images})});
    const body=await response.json();if(!response.ok)throw Error(body.error);
    for(const url of imageURLs)URL.revokeObjectURL(url);imageURLs=selected.map(f=>URL.createObjectURL(f));result=body;draw();
    const other=body.scans.filter(s=>s.cards.length&&s.cards.filter(c=>c.display_mode==='other').length>=Math.ceil(s.cards.length/2)).length;
    const unknown=body.scans.reduce((n,s)=>n+s.unidentified.length,0);
    $('status').textContent='已完成 · '+body.box.unique_count+' 张不同卡片 · '+(body.scans.reduce((n,s)=>n+s.elapsed_ms,0)/1000).toFixed(2)+' 秒'+
      (unknown?'。有 '+unknown+' 张卡片未能确认身份；如果是新卡，请更新 master 数据并重新运行 prepare 后再识别。':'')+
      (other?'。部分截图显示其他参数；要补齐等级或特训次数，请上传对应显示模式的截图。':'');
  }catch(e){$('status').textContent='识别失败：'+e.message}
  finally{$('scan').disabled=!files.length;$('reset').disabled=false}
};
$('export').onclick=()=>{
  const blob=new Blob([JSON.stringify(result.box,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='ournotes-box.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('reset').onclick=()=>{
  for(const url of imageURLs)URL.revokeObjectURL(url);imageURLs=[];files=[];result=null;$('upload').value='';setFiles([]);
  $('export').disabled=true;$('status').textContent='';for(const id of ['members','snaps','duplicates','review'])$(id).textContent='0';emptyRows('识别结果会显示在这里');
};
