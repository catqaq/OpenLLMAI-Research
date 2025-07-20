### Updates & States

- **Brief Blog**: 250709, Too occupied to proceed further, so I'll pause here for now.
- **Blog (Chinese)**: In progress
- **Blog (English)**: In progress
- **Idea Validation**: Completed, 72b/30b
- **Experiments**: In progress
- **Paper**: To be done

### Insights

There is a classic question in the LLM field: Why is RL still needed after SFT (since HF is no longer that crucial)? Is it about the exploration and exploitation of RL? The core and most important reason for transitioning from SFT to RL is that RL can learn from mistakes.

But can current mainstream RL algorithms achieve this? To draw a conclusion first, most algorithms can only partially achieve it. We know that the essential differences between various RL algorithms mainly lie in advantage estimation or baseline selection. Suppose in an extreme situation where all responses within the range of calculating advantages are wrong. In that case, most algorithms will learn nothing, while only PPO will still be effective because its baseline is obtained through critic learning prediction.

It seems that classic PPO can learn from a single error sample. Then, why not directly use PPO in extreme cases? It looks feasible, but our view is that PPO's learning from errors is also indirect and inefficient.

At the same time, there have indeed been many recent works discussing whether RL can actually improve the upper limit of models or if it is just approaching the upper limit of SFT or base model pass@N. Given the current state of mainstream RL algorithms, such doubts are not surprising, especially with the recent popularity of GRPO, which seems to be doing little more than approaching pass@N.

So how to change this? Those who have tutored children with their homework know: When a child can't solve a problem, forcing them to try repeatedly (sampling multiple times) is useless. Scientific educational methods tell us that we should guide and inspire children when they are stuck, rather than just asking them to try more. For example, provide a reference answer, tell the child they are wrong and where, and let them reflect and correct.

This leads to the core idea of this work: **Recognize mistakes and correct them**. Let the model know it's wrong and where, then let it correct, and finally learn from the corrected results, truly breaking through the limitations of the base model.

### Methods

Taking GRPO as an example (the core idea applies to other RL algorithms; GRPO is chosen only because it aligns with my initial perception when first encountering RL, emphasizing "trying more").

We know that GRPO calculates relative advantages and works well when there are both positive and negative values in the group. But what if all rewards in the group are the same? The answer is that GRPO will fail. For all 1s, it's not a big problem, but for all 0s, it's a serious issue. All 0s represent a severe error situation, but GRPO will not generate any reward (ignoring KL), which is equivalent to skipping the difficult samples with all 0s. Is this reasonable? Obviously not.

So what to do? Keep forcing the child to retry? Clearly, it's of little use. The child might go crazy, and so might we. What if the performance (reward) keeps not improving? We can't just wait; we need to tutor, tell the child where they went wrong, and let them correct. The specific process is shown in the following figure:

![image](https://xianyunlp.oss-cn-hangzhou.aliyuncs.com/uPic/image.png)

OK, now we have enabled the child to sample positive examples through tutoring. Next, how to use these positive examples? After all, they are not directly produced by the child, i.e., not on-policy. Direct use will definitely have some problems, as others' experiences cannot be directly copied. Importance sampling comes to mind, which is natural. However, in addition to the prob ratio between the old and new policies, we also introduce the prob ratio of the current policy on the old and new responses to correct the overall prob ratio. The specific formula is as follows:

$$ \mathcal{J}*{\text{GRPO}}(\theta) = \mathbb{E}\left[ q \sim P(Q), \{o_i\}*{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q) \right] \\ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} R_{\text{off}} \hat{A}*{i,t}, \text{clip} \left( \frac{\pi*\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}R_{\text{off}}, 1 - \varepsilon, 1 + \varepsilon \right) R_{\text{off}} \hat{A}*{i,t} \right] - \beta \text{KL} \left[ \pi*\theta \| \pi_{\text{ref}} \right] \right\}, $$

where $R_{\text{off}}$ is the probability ratio of the policy model on the responses before and after repair, which we call the **off-policy ratio** (In fact, this concept is not only applicable before and after correction but also in any off-policy scenario). Specifically, let $o$ and $o_f$ be the outputs before and after correction, respectively. For clarity, the subscripts $i$ and $t$ are omitted. Here, the ratio before correction is generally greater than 1 compared to that after correction, aiming to use the off-policy corrected response more conservatively.

$$ R_{\text{off}} = \begin{cases} 1, & \text{when } \text{fix} = 0 \\ \left( \dfrac{\pi_\theta(o|q)}{\pi_\theta(o_f|q)} \right)^\gamma, & \text{when } \text{fix} = 1 \end{cases} $$

**Note 1**: Since the lengths of responses before and after correction are likely different and many tokens at the same position are likely to change, it cannot be written at the token level as when the old and new policies have the same response.

**Note 2**: The reason for **choosing the correction operation is to reduce the degree of off-policy**, ensuring that the responses before and after correction are not too different. However, in practice, this approach should also be applicable to other operations, such as rewrite without reward reflection, rewrite with reward reflection, rewrite with error information, rewrite based on reference answers, etc. We will try these different operations in subsequent experiments.

More generally, we can **very conveniently introduce $R_{\text{off}}$ into various RL algorithms** by simply multiplying this coefficient, enabling all RL algorithms to have the function of recognizing and correcting mistakes.

P.S. Because I have been lazy about writing papers, I recently discovered some equally interesting works while writing. We will discuss the key differences between our work and existing works in subsequent blogs. For example:

**Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning**

### Experiments

In progress

### Results

To be presented

### Future Work

- Explore different correction operations (e.g., rewrite without reward reflection, rewrite with reward reflection, etc.)
- Conduct more comprehensive experiments to validate the effectiveness of the proposed method
- Compare with more existing works to highlight the advantages of our approach

### Contact

We welcome open-source collaboration to do something truly valuable. If you are interested, please contact us:

- Xianyu: Founder of OpenLLMAI
- Email: xianyuai@openllmai.top

### References

[1] **Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning**https://huggingface.co/papers/2505.24726

### Cite Us

Please indicate the source when reprinting: https://openllmai.notion.site/?pvs=143

Citation:

Xianyu. (July 9th, 2025.). *RFPO: Recognizing and Fixing Errors in Policy Optimization to Unlock LLMs' Potential in RL.* [Blog post]. https://openllmai.notion.site/RFRL

@online{xianyu-RFPO, title={RFPO: Recognizing and Fixing Errors in Policy Optimization to Unlock LLMs' Potential in RL.}, author={Xianyu}, year={2025}, month={July}, org={OpenLLMAI}, url={\url{https://openllmai.notion.site/?pvs=143}}, }