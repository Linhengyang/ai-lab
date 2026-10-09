# research experiments

experiments 是算法/模型开发最初始的阶段，充分验证有效性和可行性  
将完整实现写到合适的位置，比如模型结构写入aiml/，训练基建写入training/，推理基建写入inference/，其他底层构建写入native/ 等等  
在 tests 测试这些完整实现  
测试通过后将典型性能报告 写入 benchmarks  