updates & states：

简要博客：250709，实在太忙，且停笔于此；

博客：doing

英文博客：doing

Idea Validation：done，72b/30b

exps：doing

paper：TODO

# Insights：

LLM领域有个非常经典的问题：有了SFT之后为什么还需要RL（HF不是那么重要了）？

是所谓RL的探索与利用？这里最核心也是从SFT转向RL的最重要的原因是RL可以从错误中学习。

但是目前主流的RL算法能做到吗？

先说结论，绝大部分算法只能部分做到。我们知道各种RL算法的本质区别主要在于优势估计或者说baseline选择，假设一个比较极端的情况，在计算优势的response范围内大家都是错的，那大部分算法都学不到任何东西，只有PPO会依旧有效，因为它的baseline是由critic学习预测得到的。

那似乎经典的PPO是可以从单个错误样本中学习的，那我们在极端情况直接使用PPO不就好了吗？看着好像可行，但我们的观点是PPO对错误的学习也是间接的低效的。

同时，最近也确实有很多工作在讨论RL到底能否提高模型的上限，是否只是在逼近SFT或者base 模型 pass@N的上限？基于目前主流RL算法的现状，有这种怀疑其实不奇怪，尤其是最近大火的GRPO，做的事情太像在逼近 pass@N了。

那么如何改变这一切呢？

辅导过小朋友作业的大朋友们都知道：孩子不会的时候，你再逼他（采样再多次）都没用。科学的教育方法会告诉我们应该在孩子陷入困境时进行引导和启发，而非一味的要求多试试。比如提供一个参考答案，然后告诉孩子错了、哪儿错了，让孩子进行反思、修正。

由此，引出本工作核心的想法：知错就改，让模型知道错了以及知道哪里错了，然后让模型进行修正，最后学习修正后的结果，真正意义上突破基础模型的限制。

# Methods

下面以GRPO为例（其余RL算法均适用，选择GRPO仅仅是因为符合我刚接触RL时的初感知，多试试）来说明知错就改的核心思路。

我们知道GRPO计算的是相对优势，group里面有正有负时可以很好的工作，但是如果group内reward都一样呢？答案是GRPO会失效，对于全1，其实问题不大，但对于全0，问题就很严重了，全0是很严重的错误情况了，但是GRPO不会产生任何的奖励（忽略KL），相当于跳过了全0的困难样本？这合理吗？显然不合理。

那么怎么办呢？一直逼孩子重试吗？显然没啥大用，孩子不疯我们都疯了。成绩（reward）一直上不去怎么办？不能干等，你得辅导呀，告诉孩子哪儿错了，让他修改。具体流程如下图：

![image](https://xianyunlp.oss-cn-hangzhou.aliyuncs.com/uPic/image.png)

OK，现在我们通过辅导的方式让孩子采样到正例了，接下来怎么用这个正例呢？毕竟这不是孩子自己直接做出来的，即不是on policy的，直接用肯定会有些问题的，别人的经验不能直接照搬套用。大家可能会想到重要性采样，这是自然；但是除了新旧策略直接的prob ratio，这里我们还引入了当前策略在新旧response上的prob ratio来修正整体的prob ratio。具体公式如下：

$$ \mathcal{J}*{\text{GRPO}}(\theta) = \mathbb{E}\left[ q \sim P(Q), \{o_i\}*{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q) \right] \\ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} R_{\text{off}} \hat{A}*{i,t}, \text{clip} \left( \frac{\pi*\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}R_{\text{off}}, 1 - \varepsilon, 1 + \varepsilon \right) R_{\text{off}} \hat{A}*{i,t} \right] - \beta \text{KL} \left[ \pi*\theta \| \pi_{\text{ref}} \right] \right\}, $$

其中$R_{\text{off}}$是policy model在修复前后的response上的概率ratio，我们将其称之为**离线率**（**off policy ratio**，事实上这个概念不仅在修正前后可用，其实在任意off policy下都可用），具体取值如下，o和$o_{\text{f}}$分别为修正前后的输出，为了清晰，省略了i,t脚标。这里修正前比修正后的ratio一般是大于1的，目的是为了更保守的使用off policy修正的response。

$$ R_{\text{off}} = \begin{cases} 1, & \text{当 } \text{fix} = 0 \\ \left( \dfrac{\pi_\theta(o|q)}{\pi_\theta(o_f|q)} \right)^\gamma, & \text{当 } \text{fix} = 1 \end{cases} $$

注1：由于修正前后的response长度大概率不一样、很多同位置的token大概率也会发生变化，所以这里无法像新旧策略同一response时写成token level的。

注2：之所以**选择修正这个操作是出于减小off policy程度的考量**，尽量保障修正前后的response不要差的太远，但实际上这种做法对于其他操作应该也是可以的，比如不带奖励反思重写、带奖励反思重写、带错误信息重写、根据参考答案重写等，我们将在后续的实验中尝试这些不同的操作。

更一般的，我们可以**非常方便的将$R_{\text{off}}$引入到各种RL算法**中，只需要乘上这个系数即可，实现为所有RL算法添加知错就改的功能。

ps.因为一直懒得写论文，最近写论文时发现了一些同样有意思的工作，我们将在后续的博客中讨论，我们与现有工作的关键区别。比如：

**Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning**

# Exps

# Results

# Future work

# Contact

欢迎开源合作，做一些真正有价值的事情，有意者请联系我们：

- xianyu: OpenLLMAI founder
- email: xianyuai@openllmai.top

# References

[1]**Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning**https://huggingface.co/papers/2505.24726

# Cite us:

转载请注明出处：https://openllmai.notion.site/?pvs=143

引用：

xianyu. (July 9th, 2025.). 《RFPO: Recognizing and Fixing Errors in Policy Optimization to Unlock LLMs' Potential in RL.》[Blog post]. https://openllmai.notion.site/RFRL

@online{xianyu-RFPO,

title={RFPO: Recognizing and Fixing Errors in Policy Optimization to Unlock LLMs' Potential in RL.},

author={xianyu},

year={2025},

month={July},

org={OpenLLMAI},

url={\url{https://openllmai.notion.site/?pvs=143}},

}