
KITTI-360 数据准备脚本说明
==========================

文件
----
- kitti360_dataset_pipeline.py    主脚本
- kitti360_urls.example.json      下载清单示例（填官方 URL 或本地 zip 路径）
- kitti360_batch_example.json     batch-export 清单示例（10 scenes）

推荐流程
--------
1. 先去 KITTI-360 官网注册并拿到官方下载链接/下载脚本。
2. 把官方 zip 的路径或 URL 填进 `kitti360_urls.example.json`，另存为 `urls.json`。
3. 运行下载：
   python kitti360_dataset_pipeline.py download --manifest urls.json --root ./KITTI-360
4. 查看某个序列有哪些官方 fused static windows：
   python kitti360_dataset_pipeline.py list-windows --root ./KITTI-360 --sequence 0000
5. 把某个官方 static window 裁剪/下采样后重新导出为 PLY：
   python kitti360_dataset_pipeline.py prepare-fused --root ./KITTI-360 --sequence 0000 --window 0000000000_0000000240 --out ./processed/0000_clean.ply --crop "-20,80,-30,30,-2,8" --voxel-size 0.10 --visible-only
6. 一次性导出 10 个 scene（填好 kitti360_batch_example.json 后）：
   python kitti360_dataset_pipeline.py batch-export --manifest kitti360_batch_example.json --root ./KITTI-360 --out-dir ./processed --voxel-size 0.10 --visible-only
7. 如果你想从 raw velodyne 自己累积成 world-frame PLY：
   python kitti360_dataset_pipeline.py build-from-raw --root ./KITTI-360 --sequence 0000 --start 0 --end 240 --stride 5 --out ./processed/0000_raw_accum.ply --crop "-20,80,-30,30,-2,8" --voxel-size 0.10

建议
----
- 你们做 GMM + RRT，第一版优先用 `prepare-fused` 或 `batch-export`，因为官方 static window 已经是 world-frame 的 accumulated point cloud。
- 等 pipeline 跑通后，再用 `build-from-raw` 生成你们自己的窗口做 ablation。

坐标变换说明
------------
build-from-raw 使用的 velo→world 变换链：
  T_{world←velo} = cam0_to_world(frame) @ inv(calib_cam_to_velo)

cam0_to_world.txt 和 calib_cam_to_velo.txt 都定义在 unrectified cam0 坐标系下，
所以两者直接拼接即可，R_rect_00 不参与此链（R_rect 只在投影到图像时才需要）。

如果某帧没有精确匹配的 pose，脚本会用 SE(3) slerp（四元数球面插值 + 平移线性插值）
在相邻两帧 pose 之间做插值。
