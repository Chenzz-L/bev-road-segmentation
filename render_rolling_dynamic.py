"""Actual-frame replay: live road maps, current candidates, +2s occupancy, ego path."""
import base64,io,json,shutil,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import rolling_dynamic_road as run
from dynamic_road_core import ROI

def main():
    out=run.OUT;verify=run.read(out/'verification.json');assert verify['status']=='PASS'
    rows=run.read(out/'manifest.json');objects=run.read(out/'rolling_objects.json');protocol=run.read(out/'protocol.json')
    with np.load(out/'rolling_outputs.npz') as z:data={k:z[k] for k in z.files}
    fontpath=os.environ.get('ROAD_CJK_FONT', '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc')
    fonts={n:ImageFont.truetype(fontpath,n) for n in (15,17,19,23,27)}
    poses=data['ego_to_global'];origin=poses[0,:2,3];path=poses[:,:2,3]-origin
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    lo=path.min(0)-32;hi=path.max(0)+32;mid=(lo+hi)/2;scale=min(590/(hi[0]-lo[0]),230/(hi[1]-lo[1]))
    def global_pixel(xy):
        v=(np.asarray(xy)-origin-mid)*scale
        return (int(940+v[0]),int(250-v[1]))
    def raster(a):return Image.fromarray(np.flip(a.transpose(1,0,2),(0,1)).copy()).resize((350,350),Image.Resampling.NEAREST)
    images=[];thumbs=[];stats=[]
    for i,row in enumerate(rows):
        image=Image.new('RGB',(1280,880),'#f4f7fb');d=ImageDraw.Draw(image)
        elapsed=row['elapsed_s'];speed_label='—' if i==0 else '%.1f m/s'%((distance[i]-distance[i-1])/(elapsed-rows[i-1]['elapsed_s']))
        d.text((20,10),'连续逐帧道路感知与动态占据回放',font=fonts[27],fill='#142b47')
        d.text((20,48),'%s  |  帧 %02d / %02d  |  t = %.2f s  |  自车累计行驶 %.1f m  |  观测速度 %s'%(run.SCENE,i+1,len(rows),elapsed,distance[i],speed_label),font=fonts[17],fill='#415a76')
        with Image.open(row['images']['CAM_FRONT']['img_path']) as camera:camera=ImageOps.contain(camera.convert('RGB'),(580,326))
        image.paste(camera,(20+(580-camera.width)//2,85+(326-camera.height)//2))
        d.rounded_rectangle((620,85,1260,411),radius=8,fill='#142437');d.text((634,95),'全局视图：自车已行驶轨迹（非未来规划）',font=fonts[17],fill='white')
        if i>0:d.line([global_pixel(x+origin) for x in path[:i+1]],fill='#7ae8c3',width=4)
        start=global_pixel(origin);d.ellipse((start[0]-4,start[1]-4,start[0]+4,start[1]+4),fill='#b5bdcb')
        e=poses[i];corners=np.array([[-24,-24],[24,-24],[24,24],[-24,24],[-24,-24]])@e[:2,:2].T+e[:2,3]
        d.line([global_pixel(c) for c in corners],fill='#4484be',width=2)
        for obj in objects[i]['objects']:
            xy=np.array(obj['origin'])
            if np.linalg.norm(xy-e[:2,3])>35 or obj['score']<.3:continue
            pt=global_pixel(xy);d.ellipse((pt[0]-2,pt[1]-2,pt[0]+2,pt[1]+2),fill='#f2a84b')
            d.line([global_pixel(x) for x in obj['future_global'][:5]],fill='#ac753d',width=1)
        pos=global_pixel(e[:2,3]);head=global_pixel(e[:2,3]+e[:2,0]*6)
        d.ellipse((pos[0]-5,pos[1]-5,pos[0]+5,pos[1]+5),fill='#ffe275');d.line((pos,head),fill='#ffe275',width=3)
        d.text((635,380),'黄点：自车    蓝框：当前 BEV 范围    橙点：当前车辆',font=fonts[15],fill='#d7e6f5')
        p=data['probability'][i];road=p>=protocol['road_threshold'];ground=data['ground'][i];blocked=data['blocked'][i];candidate=road&ground&~blocked&ROI;future=data['occupancy'][i,2]
        low=np.array([23,41,64]);high=np.array([112,229,178]);heat=(low+(high-low)*p[:,:,None]).astype(np.uint8)
        current=np.full((96,96,3),[23,41,64],np.uint8);current[road]=[80,104,125];current[candidate]=[87,201,160];current[blocked]=[156,163,174]
        dynamic=current.copy();dynamic[future]=[222,157,74];dynamic[future&candidate]=[249,104,66]
        for x,a,title in [(30,heat,'① 当前帧道路概率（每帧重新推理）'),(465,current,'② 当前点云与障碍物过滤'),(900,dynamic,'③ 从当前帧预测未来 2 秒占据')]:
            image.paste(raster(a),(x,460));d.text((x,426),title,font=fonts[17],fill='#142b47')
            d.polygon([(x+175,625),(x+169,639),(x+181,639)],fill='white')
        d.text((22,824),'颜色：亮绿为高道路概率；绿色为放行候选；灰色为当前障碍；橙色为未来占据；红色为候选区与未来占据重叠。',font=fonts[17],fill='#24405c')
        d.text((22,851),'下方为每帧自车坐标系：上方朝前，左方朝左。约 2 Hz 离线关键帧回放，无补帧；道路头没有时序平滑。',font=fonts[15],fill='#52677c')
        stream=io.BytesIO();image.save(stream,format='JPEG',quality=86);images.append(base64.b64encode(stream.getvalue()).decode('ascii'))
        thumbs.append(image.resize((960,660),Image.Resampling.LANCZOS))
        if i in (0,20,39):image.save(out/('frame_%02d.png'%i))
        stats.append(dict(frame=i,elapsed_s=elapsed,ego_distance_m=float(distance[i]),road_cells=int(road.sum()),candidate_cells=int(candidate.sum()),future_2s_intrusion=int((future&candidate).sum())))
    durations=[max(100,round((rows[i+1]['elapsed_s']-rows[i]['elapsed_s'])*1000)) for i in range(len(rows)-1)]+[500]
    thumbs[0].save(out/'rolling_replay.gif',save_all=True,append_images=thumbs[1:],duration=durations,loop=0,optimize=True)
    js='const frames='+json.dumps(images)+',times='+json.dumps([r['elapsed_s'] for r in rows])+''';
const image=document.getElementById('frame'),slider=document.getElementById('slider'),label=document.getElementById('time'),button=document.getElementById('play'),speed=document.getElementById('speed');let timer=null,running=false;
function draw(){let i=+slider.value;image.src='data:image/jpeg;base64,'+frames[i];label.textContent='帧 '+(i+1)+' / '+frames.length+' · '+times[i].toFixed(2)+' s'}
function tick(){if(!running)return;let i=+slider.value,delay=(i+1<times.length?times[i+1]-times[i]:.5)*1000/(+speed.value);timer=setTimeout(()=>{slider.value=(i+1)%frames.length;draw();tick()},delay)}
button.onclick=()=>{running=!running;button.textContent=running?'暂停':'播放';clearTimeout(timer);if(running)tick()};slider.oninput=()=>{clearTimeout(timer);draw();if(running)tick()};speed.onchange=()=>{clearTimeout(timer);if(running)tick()};draw();
'''
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连续道路感知与动态占据回放</title><style>body{max-width:1300px;margin:24px auto;padding:16px;background:#f4f7fb;font:16px/1.7 system-ui;color:#19324c}img{width:100%;border-radius:12px}button,select{font:inherit;padding:6px 14px;margin:8px}input{width:40%;vertical-align:middle}p{margin:12px 0}a{color:#1369af}</style><h1>连续道路感知与动态占据回放</h1><p>40 个真实连续关键帧，19.50 秒，自车行驶 128.52 米。道路预测、当前地面/障碍物与未来占据随当前帧同步更新。未来预测时域固定为 2 秒，滑块改变的是实际采集时间。</p><button id="play">播放</button><input id="slider" type="range" min="0" max="39" value="0"><span id="time"></span><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select><img id="frame" alt="同步前视图、全局自车轨迹与三种 BEV 输出"><p>左下道路概率来自每帧新提取的 BEVFusion 特征；中间候选区使用该帧真实点云；右下使用对应时间戳的既有因果跟踪/GRU 输出。全局视图只画已观测的自车轨迹，不是自车未来路径规划。</p><p>本次没有训练新模型，也没有改成时序 BEV 网络。它是约 2 Hz 的离线逐帧回放，不能据此声称端到端实时运行。预测的动态对象仍限于小汽车。</p><p><a href="rolling_replay.gif">下载 GIF 回放</a> · <a href="verification.json">推理与一致性核查</a> · <a href="fresh_inference.json">逐帧特征和概率指纹</a> · <a href="protocol.json">方案与来源</a></p><script>'''+js+'</script></html>'
    (out/'report.html').write_text(doc,encoding='utf-8');run.save('frame_statistics.json',stats)
    for n in ('rolling_dynamic_road.py','render_rolling_dynamic.py'):shutil.copy2(run.ROOT/n,out/n)
    run.save('artifacts.json',{f.name:dict(bytes=f.stat().st_size,sha256=run.sha(f)) for f in out.iterdir() if f.is_file() and f.name!='artifacts.json'})
    print('ROLLING_RENDERED',len(images),flush=True)

if __name__=='__main__':main()
