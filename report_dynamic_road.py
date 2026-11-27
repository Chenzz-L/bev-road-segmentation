"""Create a self-contained horizon viewer and archive the no-training prototype."""
import base64,html,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import dynamic_road_trial as run
from dynamic_road_core import METHODS,ROI,release

OUT=run.OUT
def enc(a):return base64.b64encode(np.packbits(a).tobytes()).decode('ascii')
def main():
    summary=run.read(OUT/'summary.json');protocol=run.read(OUT/'protocol.json');assert run.read(OUT/'verification.json')['status']=='PASS'
    refs=run.refdata();p=np.load(run.BASE/'val_wide_fusion_17_prob.npy',mmap_mode='r');threshold=protocol['road_thresholds']['17']
    with np.load(OUT/'prediction_masks.npz') as z:pred={k:z[k] for k in z.files}
    with np.load(OUT/'evaluation_truth.npz') as z:truth=z['masks']
    demos=run.read(OUT/'demo_selection.json');cases=[];frames=[]
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),19) if fontpath.exists() else ImageFont.load_default()
    small=ImageFont.truetype(str(fontpath),14) if fontpath.exists() else ImageFont.load_default()
    for di,item in enumerate(demos):
        i=item['index'];candidate=release(p[i],threshold,refs['ground'][i],refs['blocked'][i],np.zeros((96,96),bool));road=p[i]>=threshold
        cases.append(dict(scene=item['scene'],index=i,road=enc(road),candidate=enc(candidate),blocked=enc(refs['blocked'][i]&ROI),ground=enc(refs['ground'][i]),valid=pred['valid'][i].tolist(),times=pred['dt'][i].tolist(),occupancy=[[enc(v) for v in h] for h in pred['masks'][i]],truth=[enc(v[0]) for v in truth[i]],selection_rule='GRU predicted intrusion, no GT-based selection'))
        if di==0:
            for hi,h in enumerate(run.HORIZONS):
                canvas=Image.new('RGB',(1140,490),'#f7f9fc');draw=ImageDraw.Draw(canvas)
                draw.text((20,12),'Future occupancy preview | '+item['scene']+' | t + %.0f s'%h,font=font,fill='#14233b')
                for panel,(method,title) in enumerate([('hold','Current boxes held'),('ctrv','CTRV motion'),('gru','GRU motion')]):
                    occ=pred['masks'][i,hi,METHODS.index(method)];a=np.full((96,96,3),[30,42,57],np.uint8)
                    a[road]=[80,94,107];a[candidate]=[91,185,153];a[refs['blocked'][i]&ROI]=[141,150,161]
                    a[occ]=[231,156,59];a[occ&candidate]=[245,113,68]
                    # Blue boundary is evaluation-only truth; never fed into the predictor.
                    gt=truth[i,hi,0];interior=gt.copy()
                    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:interior &= np.roll(np.roll(gt,dy,0),dx,1)
                    a[gt&~interior]=[74,182,255]
                    img=Image.fromarray(np.flip(a.transpose(1,0,2),(0,1)).copy()).resize((350,350),Image.Resampling.NEAREST)
                    x=20+panel*375;canvas.paste(img,(x,75));draw.text((x,48),title,font=small,fill='#14233b')
                    draw.line((x+175,260,x+175,244),fill='white',width=3);draw.polygon([(x+175,239),(x+169,248),(x+181,248)],fill='white')
                draw.text((20,442),'Green: current release candidate   Orange: forecast car occupancy   Blue outline: future GT (evaluation only)',font=small,fill='#14233b')
                draw.text((20,465),'Fixed current ego frame. Up = forward, left = ego left. No newly freed occluded area.',font=small,fill='#14233b')
                frames.append(canvas)
    frames[-1].save(OUT/'preview_3s.png');frames[0].save(OUT/'horizon_preview.gif',save_all=True,append_images=frames[1:],duration=1100,loop=0)
    rows=[]
    for h in ('1','2','3'):
        v=summary['horizons'][h]
        for a in ('hold','cv','ctrv','gru','gru_pad'):
            m=v['methods'][a];rows.append('<tr><td>%s s</td><td>%s</td><td>%d</td><td>%.1f</td><td>%.2f%%</td><td>%.2f%%</td></tr>'%(h,a,v['frames'],m['car_conflict'],100*m['coverage'],100*m['occupancy_recall']))
    script=r'''
const cases=DATA,methods=METHODDATA;const $=id=>document.getElementById(id);
function unpack(s){let b=atob(s),a=new Uint8Array(9216);for(let i=0;i<a.length;i++)a[i]=(b.charCodeAt(i>>3)>>(7-(i&7)))&1;return a}
cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.text=c.scene;$('case').add(o)});
methods.forEach((m,i)=>{let o=document.createElement('option');o.value=i;o.text=m;$('method').add(o)});$('method').value=3;
function paint(canvas,c,occ,showTruth,h){let ctx=canvas.getContext('2d'),a=ctx.createImageData(96,96),road=unpack(c.road),base=unpack(c.candidate),blocked=unpack(c.blocked),gt=unpack(c.truth[h]);
for(let y=0;y<96;y++)for(let x=0;x<96;x++){let k=y*96+x,o=((95-x)*96+95-y)*4,col=[30,42,57];if(road[k])col=[80,94,107];if(base[k])col=[91,185,153];if(blocked[k])col=[141,150,161];if(occ&&occ[k])col=base[k]?[245,113,68]:[231,156,59];if(showTruth&&gt[k]&&(!gt[k-1]||!gt[k+1]||!gt[k-96]||!gt[k+96]))col=[74,182,255];a.data.set([...col,255],o)}ctx.putImageData(a,0,0);ctx.fillStyle='white';ctx.beginPath();ctx.moveTo(48,45);ctx.lineTo(46,50);ctx.lineTo(50,50);ctx.fill();}
function update(){let c=cases[+$('case').value],h=+$('time').value,m=+$('method').value;$('t').textContent=['当前','约 1 秒后','约 2 秒后','约 3 秒后'][h]+'（实际 '+c.times[h].toFixed(3)+' s）';paint($('current'),c,null,false,h);paint($('forecast'),c,unpack(c.occupancy[h][m]),$('gt').checked,h);$('note').textContent='选择方式：根据模型预测的未来侵入面积选取示例，不根据真实标签或改善幅度选例。深色未知区域、当前障碍物区域不会因为预测车辆离开而自动放行。';}
['case','method','time','gt'].forEach(x=>$(x).addEventListener('input',update));let timer=null;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放预测时域'}else{timer=setInterval(()=>{$('time').value=(+$('time').value+1)%4;update()},1100);$('play').textContent='暂停'}};update();
'''.replace('DATA',json.dumps(cases,ensure_ascii=False),1).replace('METHODDATA',json.dumps(METHODS),1)
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BEV 动态占据与放行候选原型</title><style>body{max-width:1120px;margin:30px auto;padding:20px;font:16px/1.65 system-ui;background:#f5f7fb;color:#17283d}section{background:white;padding:20px;border-radius:14px;margin:18px 0}canvas{width:100%;max-width:430px;image-rendering:pixelated}.panels{display:flex;gap:30px;flex-wrap:wrap}.panels>div{flex:1;min-width:280px}select,button,input{font:inherit;margin:8px;padding:6px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #d4dce8;text-align:right}td:first-child,th:first-child{text-align:left}.note{color:#465a73}pre{white-space:pre-wrap}a{color:#1465b4}</style><h1>BEV 动态占据与放行候选原型</h1><p>复用已有道路头、雷达跟踪与 GRU；没有新训练。第一版只提前限制可能被小汽车占据的当前候选区域。</p><section><label>示例<select id="case"></select></label><label>运动模型<select id="method"></select></label><br><button id="play">播放预测时域</button><input id="time" type="range" min="0" max="3" value="3"><b id="t"></b><label><input type="checkbox" id="gt">显示未来真实车辆轮廓（仅评估）</label><div class="panels"><div><h3>当前道路候选与障碍物</h3><canvas id="current" width="96" height="96"></canvas></div><div><h3>预测时刻的车辆占据</h3><canvas id="forecast" width="96" height="96"></canvas></div></div><p>绿色：当前可放行候选；橙色：预计车辆占据；橙红：对当前候选区的新增拦截；灰色：当前障碍物；深色：未知。上方为车辆前方，左方为车辆左侧，始终使用当前帧坐标系。</p><p class="note" id="note"></p></section><section><h2>完整评估</h2><p>原始 150 场景、542 帧；按各时域未来帧是否存在进行客观筛选。保留没有检测、匹配失败、新进入场景的真实车辆。表中为三个道路头种子的平均值，GRU 固定使用原有模型；冲突单位为栅格观测次数，不是碰撞次数。</p><table><tr><th>预测时域</th><th>模型</th><th>帧数</th><th>放行与未来真实车辆冲突栅格</th><th>可评估道路覆盖</th><th>全网格车辆占据召回</th></tr>'''+''.join(rows)+'''</table><p>hold：当前车辆保持原位；cv：匀速；ctrv：恒转率；gru：已有残差 GRU；_pad：每边增加 0.25 + 0.25 × 实际秒数（米）的固定敏感性对照，未进行置信度校准。</p><p>可评估道路覆盖的分母为：地图道路 ∩ 当前地面证据 ∩ 当前未阻挡 ∩ 未来没有已标注对象。它不代表完整未来道路覆盖或物理安全。</p></section><section><h2>结果比较与边界</h2><pre>'''+html.escape(json.dumps({h:summary['horizons'][h]['comparisons'] for h in ('1','2','3')},ensure_ascii=False,indent=2))+'''</pre><ul>'''+''.join('<li>'+html.escape(x)+'</li>' for x in summary['limits'])+'''</ul><p><a href="summary.json">完整指标</a> · <a href="protocol.json">锁定方案</a> · <a href="verification.json">核查记录</a></p></section><script>'''+script+'</script></html>'
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    for name in ('dynamic_road_core.py','dynamic_road_trial.py','report_dynamic_road.py','verify_dynamic_road.py'):shutil.copy2(run.ROOT/name,OUT/name)
    checks={p.name:dict(sha256=run.sha(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file() and p.name!='artifacts.json'}
    run.save('artifacts.json',dict(files=checks,status='COMPLETE',no_new_training=True))
    print('REPORT_DONE',flush=True)

if __name__=='__main__':main()
