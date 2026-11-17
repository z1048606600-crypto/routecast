# 验证范围

本地验证环境为 Windows x64、Python 3.12、CPU；依赖安装在工作区独立目录，未改动系统 Python。

已完成：核心训练 / 评估脚本、API TestClient 集成验证、相应 pytest 测试。两个 PyTorch 项目完成 ONNX Runtime batch=1 和 4 的数值一致性检查。

未完成：真实业务数据验证、生产设备接入、Docker 镜像构建、GitHub Actions 云端运行、真实本地大模型推理。可选 YOLO 权重后端仅提供适配代码，不列为已实测。

运行时数值、样例和来源标记保留在 `artifacts/`。测试结果汇总见作品集根目录验证报告。

代码与报告是原型交付物。履历应写个人项目和实际贡献，不应包装为所在单位生产系统。
