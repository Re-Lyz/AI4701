**1. 为什么池化能实现对输入中“小幅平移”的近似不变性？**
 池化层在 k × k 的局部窗口内对特征做汇聚（取最大值或平均值），只关心“窗口里出现了什么”，而不关心它精确落在第几行第几列。当输入图像或特征图发生小于窗口/步幅的位移时，落入同一窗口的像素集合几乎不变，因此池化输出保持一致或仅有极小变化。从信号角度看，这相当于一次低通滤波加下采样：高频位置差异被过滤掉，输出对微小平移天然不敏感。卷积层本身满足“平移等变”，再经过池化的局部无关化后，就把“等变”进一步转化为“近似不变”。

**2. 这种平移不变性能带来什么好处？**
 近似平移不变性让网络专注于“特征是什么”而不是“在哪里”，带来多重优势：① **提高泛化**——模型不用为相同特征在不同位置各学一套权重，降低过拟合、减少参数；② **增强鲁棒性**——对输入的轻微对齐误差、摄像抖动或手写体笔画偏移仍能正确识别；③ **扩大感受野**——逐级池化让更高层卷积看到更大上下文，利于全图级语义判别；④ **节省计算与显存**——下采样缩小特征图，后续层运算量随之减少；⑤ **梯度更稳定**——最大池化稀疏激活、平均池化平滑信号，都有助于缓解梯度对噪声的敏感性。这些优点共同促成了卷积神经网络在分类等位置无关任务中的高效表现。