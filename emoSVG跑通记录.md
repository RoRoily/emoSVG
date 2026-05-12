# emoSVG跑通记录

### 1. 创建隔离环境+下载必要的包

略



### 2. 拉取git

服务器clone只有20kb每秒，本地网页上download zip再scp传到服务器了...

- https://github.com/facebookresearch/segment-anything                     

  - https://github.com/VAST-AI-Research/TripoSR
  - https://github.com/KwaiVGI/LivePortrait                                  
  - https://github.com/ToonCrafter/ToonCrafter  

解压并作为环境依赖

````
for zip in segment-anything-main.zip TripoSR-main.zip LivePortrait-main.zip ToonCrafter-main.zip; do name="${zip%-main.zip}"; unzip -q "$zip" -d "/tmp/$name"; pip install "/tmp/$name/${name}-main"; done
````

只有segment-everthing可以直接安装，其他需要手动

 三个都只有 requirements.txt，没有包结构，需要直接安装依赖并把源码路径加到
  Python 路径里。
                                                                             
  #安装各自的依赖

 ````
  pip install -r /tmp/TripoSR/TripoSR-main/requirements.txt                  
   pip install -r /tmp/LivePortrait/LivePortrait-main/requirements.txt
   pip install -r /tmp/ToonCrafter/ToonCrafter-main/requirements.txt
   grep -v "gradio" /tmp/TripoSR/TripoSR-main/requirements.txt | pip install
   -r /dev/stdin
 ````

````
python -c "import segment_anything; print('segment_anything OK'); import tsr; print('TripoSR OK'); from liveportrait.live_portrait_pipeline import LivePortraitPipeline; print('LivePortrait OK'); import lvdm; print('ToonCrafter OK')"
````



把源码目录加到 site-packages（相当于 pip install -e）

````
echo "/tmp/TripoSR/TripoSR-main" > $(python -c "import site;
  print(site.getsitepackages()[0])")/emosvg_triposr.pth
  echo "/tmp/LivePortrait/LivePortrait-main/src" > $(python -c "import site;
  print(site.getsitepackages()[0])")/emosvg_liveportrait.pth
  echo "/tmp/ToonCrafter/ToonCrafter-main" > $(python -c "import site;
  print(site.getsitepackages()[0])")/emosvg_tooncrafter.pth 
````

  不过注意：/tmp目录重启后会被清空。建议把这三个目录移到持久路径：

  ````
  mkdir -p ~/workspace/third_party
    cp -r /tmp/TripoSR/TripoSR-main ~/workspace/third_party/TripoSR
    cp -r /tmp/LivePortrait/LivePortrait-main
    ~/workspace/third_party/LivePortrait
    cp -r /tmp/ToonCrafter/ToonCrafter-main ~/workspace/third_party/ToonCrafter
  ````

然后 .pth 指向持久路径

````
  echo "/data3/zhengmuhan/workspace/third_party/TripoSR" > $(python -c
  "import site; print(site.getsitepackages()[0])")/emosvg_triposr.pth
  echo "/data3/zhengmuhan/workspace/third_party/LivePortrait/src" > $(python
  -c "import site; print(site.getsitepackages()[0])")/emosvg_liveportrait.pth
  echo "/data3/zhengmuhan/workspace/third_party/ToonCrafter" > $(python -c
  "import site; print(site.getsitepackages()[0])")/emosvg_tooncrafter.pth
````

