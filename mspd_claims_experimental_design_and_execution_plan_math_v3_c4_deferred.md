# MSPD: экспериментальный дизайн и compute-aware план

> Версия с Markdown-math: формулы оформлены через `$...$` и `$$...$$`. Она должна нормально рендериться в GitHub, Obsidian, Typora, VS Code Markdown Preview с math extension и большинстве современных Markdown-renderer-ов.

Документ фиксирует, как проверять шесть основных claims по MSPD с учётом того, что уже есть в текущем сетапе: группы Flow-Lenia checkpoints, A/B/C simulations для frustration assay, несколько NN-OEE exemplars, Particle Life++ runs, оптимизируемый lag $\tau$, и невозможность надёжно делать object/cell tracking или semantic color-based species detection.

Главный принцип: не строить доказательство на детекции клеток, connected components, цветовых species labels или handcrafted patch motifs из rendered video. Эти подходы ломаются на слипании, multi-scale объектах, активном субстрате, мигании цвета и одной большой связной компоненте. Экспериментальная часть должна проверять более высокоуровневые свойства: trajectory-distributional heterogeneity, transition sensitivity, scale selection, blockwise frustration, substrate transfer. C4 ecological richness is deferred because no robust species ontology is currently available.

Также не предлагается делать exhaustive robustness по каждому гиперпараметру. Все сравнения ниже используют фиксированный protocol и несколько общих наборов симуляций, чтобы одни и те же данные закрывали несколько claims.

---

## 0. Общая нотация и базовые observables

Пусть $\theta$ — параметры симуляционной системы. Для Flow-Lenia это checkpoint параметров. Для Particle Life++ это checkpoint параметров взаимодействия, включая neural interaction rule. Пусть $s$ — random seed начального состояния.

Состояние симуляции обозначается

$$
X_{\theta,s}(t).
$$

Если доступны tracked particles/tracers, их координаты обозначаются

$$
x_i(t) \in \mathbb R^2,
$$

где $i$ — индекс частицы/tracer-а.

Для лага $\tau$ и observation window $W=[t_0,t_1]$ у частицы $i$ строится эмпирическое распределение скоростей или displacement-векторов:

$$
p_{i,W}^{(\tau)}=
\frac{1}{|T_W|}
\sum_{t\in T_W}
\delta_{u_i^{(\tau)}(t)},
\qquad
u_i^{(\tau)}(t)=\frac{x_i(t+\tau)-x_i(t)}{\tau}.
$$

Здесь $T_W$ — sampled time points внутри окна $W$, для которых $t+\tau\in W$. Можно использовать displacement без деления на $\tau$, если именно displacement входит в текущий MSPD код. Важно только, чтобы одинаковая версия использовалась во всех сравнениях.

Попарное расстояние между trajectory distributions:

$$
D_{ij}^{(\tau,W)}=
\operatorname{SW}_1\left(p_{i,W}^{(\tau)},p_{j,W}^{(\tau)}\right),
$$

где $\operatorname{SW}_1$ — sliced Wasserstein-1 distance.

Локальная heterogeneity в окне:

$$
\Delta H(W,\tau)=
\frac{2}{n(n-1)}
\sum_{i<j}
D_{ij}^{(\tau,W)}-
H_0(W,\tau).
$$

$H_0(W,\tau)$ — baseline из текущей реализации: shuffled/null baseline, global baseline или другой baseline, который уже используется в MSPD. Если текущий код использует немного другую формулу, в статье надо писать именно кодовую версию. Здесь важен экспериментальный protocol, а не точная нормализация.

Scalar MSPD score для checkpoint-а:

$$
D(\theta,\tau;s,\mathcal W)=
\operatorname{Agg}_{W\in\mathcal W}\Delta H(W,\tau),
$$

где $\operatorname{Agg}$ — median или mean по windows. Для robustness лучше использовать median, потому что $\Delta H$ может иметь редкие крупные spikes.

Так как $\tau$ оптимизируется, честный comparison должен давать random controls такое же право выбирать $\tau$, как optimized checkpoints.

Для каждого checkpoint-а определяем selection split и evaluation split:

$$
\mathcal W_{\mathrm{sel}} \cap \mathcal W_{\mathrm{eval}}=\varnothing.
$$

Например, если последние 200k steps разбиты на $m$ окон, то нечётные окна идут в $\mathcal W_{\mathrm{sel}}$, чётные — в $\mathcal W_{\mathrm{eval}}$. Если есть несколько seeds, один seed можно использовать для selection, другой для evaluation. Если seeds мало, window split дешевле и применим к уже имеющимся trajectories.

Выбранный lag:

$$
\tau^\star(\theta)=
\arg\max_{\tau\in\mathcal T}
D(\theta,\tau;s_{\mathrm{sel}},\mathcal W_{\mathrm{sel}}).
$$

Final reported score:

$$
D_{\mathrm{eval}}(\theta)=
D(\theta,\tau^\star(\theta);s_{\mathrm{eval}},\mathcal W_{\mathrm{eval}}).
$$

Если optimization уже сохранила learned $\tau$, его нужно показывать отдельно как $\tau_{\mathrm{opt}}$, но primary statistical comparison лучше делать через одинаковую post-hoc selection rule $\tau^\star(\theta)$ для optimized и random checkpoints. Это убирает selection bias: random systems тоже получают возможность выбрать свой лучший temporal scale.

---

# 1. Claim C1: optimized systems vs random controls

## Исходная гипотеза

MSPD отделяет optimized Flow-Lenia systems от random controls. Random systems дают near-zero MSPD, optimized systems дают более высокие $\Delta H$, cross-window variance и scalar complexity $D_P$.

## Предлагаемая формулировка

MSPD optimization robustly increases trajectory-distributional heterogeneity relative to matched random controls under a selection-adjusted temporal-scale protocol. The increase is measured on held-out windows or held-out seeds after allowing every checkpoint, optimized and random, to select its own $\tau^\star$ by the same rule.

Эта формулировка сильнее исходной, потому что не просто говорит “мы оптимизировали метрику и она выросла”, а закрывает два главных objection-а: optimized checkpoints не получают нечестного преимущества от выбора $\tau$, и результат измеряется не на тех же окнах, на которых выбирался $\tau$.

## Эксперимент C1. Selection-adjusted optimized-vs-random comparison

### Зачем нужен эксперимент

Это основной quantitative proof, что MSPD optimization реально меняет dynamics, а не просто находит красивый единичный пример. Без этого claim C1 остаётся иллюстративным.

### Что будет, если его не провести

Reviewer может сказать, что сравнение optimized vs random нечестное, потому что optimized system получила оптимизированный $\tau$, а random baseline нет. Также reviewer может сказать, что score завышен из-за выбора окна или seed-а.

### Что ожидается увидеть

Для большинства matched groups:

$$
D_{\mathrm{eval}}(\theta_r^{\mathrm{opt}})>
\operatorname{median}_{j=1}^3
D_{\mathrm{eval}}(\theta_{r,j}^{\mathrm{rand}}).
$$

Групповой contrast:

$$
\Delta_r^{C1}=
D_{\mathrm{eval}}(\theta_r^{\mathrm{opt}})-
\operatorname{median}_{j=1}^3D_{\mathrm{eval}}(\theta_{r,j}^{\mathrm{rand}})
$$

должен быть положительным для большинства $r$. Статистически: one-sided paired sign test или Wilcoxon signed-rank test по $\Delta_r^{C1}$. Если $n$ маленькое, лучше репортить не только $p$-value, но и число положительных групп, median effect и bootstrap confidence interval.

### Подробный protocol

Для каждого Flow-Lenia group $r=1,\dots,G_{FL}$ есть один optimized checkpoint

$$
\theta_r^{\mathrm{opt}}
$$

и три matched random checkpoints

$$
\theta_{r,1}^{\mathrm{rand}},\theta_{r,2}^{\mathrm{rand}},\theta_{r,3}^{\mathrm{rand}}.
$$

Для каждого checkpoint-а используется normal simulation A и, если нужно, additional seed simulation C. Из последних 200k steps формируются windows:

$$
\mathcal W=\{W_1,\dots,W_m\}.
$$

Для каждого $\tau\in\mathcal T$ считаются $\Delta H(W_k,\tau)$ и

$$
D(\theta,\tau;s,\mathcal W_{\mathrm{sel}})=
\operatorname{median}_{W\in\mathcal W_{\mathrm{sel}}}\Delta H(W,\tau).
$$

Потом выбирается

$$
\tau^\star(\theta)=\arg\max_{\tau\in\mathcal T}D(\theta,\tau;s,\mathcal W_{\mathrm{sel}}).
$$

Evaluation score:

$$
D_{\mathrm{eval}}(\theta)=
\operatorname{median}_{W\in\mathcal W_{\mathrm{eval}}}
\Delta H(W,\tau^\star(\theta)).
$$

Если available simulation C с другим seed, можно сделать более строгую версию:

$$
\tau^\star(\theta)=\arg\max_{\tau\in\mathcal T}D(\theta,
\tau;s_A,\mathcal W_A),
$$

$$
D_{\mathrm{eval}}(\theta)=D(\theta,
\tau^\star(\theta);s_C,
\mathcal W_C).
$$

Это лучше, но требует, чтобы MSPD trajectories были посчитаны для C.

### Что рисовать

Figure C1 должна состоять из трёх панелей.

Первая панель: MSPD profile по $\tau$. По оси $x$: $\log \tau$. По оси $y$: $D(\theta,\tau;s,\mathcal W_{\mathrm{sel}})$. Тонкие линии — отдельные checkpoints, жирные линии — median по optimized и random groups. Вертикальные маркеры — selected $\tau^\star$. Это показывает, что optimized systems не выигрывают только из-за одного hand-picked temporal scale.

Вторая панель: paired contrast. Для каждой group $r$ точка $\Delta_r^{C1}$. Можно нарисовать zero line, median и confidence interval. Если хочется более наглядно: рядом dots для optimized checkpoint и median random controls, соединённые линией внутри group.

Третья панель: heatmaps $\Delta H(W,\tau)$ для representative random, representative MSPD-opt и NN-OEE exemplar. NN-OEE здесь не статистический baseline, а reference exemplar.

---

# 2. Claim C2: $\Delta H$ and ecological transitions

## Исходная гипотеза

$\Delta H(t)$ spikes correspond to species-turnover events: metric tracks ecological events rather than noise.

## Предлагаемая формулировка

Peaks of $\Delta H(t)$ identify transition-sensitive periods in the simulation: times at which local trajectory distributions become heterogeneous and future dynamics become more sensitive to small perturbations. In simulations where species identity can be independently validated visually or by a reliable auxiliary representation, these periods often coincide with ecological turnover.

Эта формулировка сохраняет смысл, но убирает уязвимость “species не определены формально”. Мы не утверждаем, что every $\Delta H$ spike is species turnover. Мы утверждаем, что $\Delta H$ выбирает transition-sensitive states, а ecological interpretation даётся в validated case studies.

## Эксперимент C2-A. Event-aligned visual case study on validated ecology

### Зачем нужен эксперимент

Нужен bridge между математической метрикой и биологической интерпретацией. Он показывает, что на конкретной сложной симуляции $\Delta H$ пики совпадают с человечески интерпретируемыми ecological transitions.

### Что будет, если его не провести

Тогда C2 будет чисто динамическим claim-ом: “$\Delta H$ ловит transition sensitivity”. Это корректно, но слабее для статьи, потому что не объясняет, почему эта метрика говорит что-то про ecology.

### Что ожидается увидеть

На выбранной validated simulation high $\Delta H(t)$ пики должны находиться около моментов, где визуально видно: один class организмов исчезает, другой становится dominant, появляются/исчезают coherent bodies, меняется interaction regime. Если в этой конкретной симуляции цвет действительно соответствует species-like roles, можно дополнительно показать color-mass derivative correlation, но только как auxiliary validation, не как основной метод.

### Подробный protocol

Выбрать одну flagship simulation, где визуально подтверждено, что цветовые компоненты действительно stable over object lifetime и связаны с ecological roles. Для этой simulation берётся time series

$$
\Delta H(t)=\Delta H(W_t,\tau^\star),
$$

где $W_t=[t-w/2,t+w/2]$, а $\tau^\star$ — selected MSPD lag для этой simulation.

Если используется color-mass auxiliary analysis, то для каждого color cluster $c$ определяется mass trajectory

$$
M_c(t)=\sum_{r}A(r,t)\mathbf 1[\operatorname{color}(r,t)=c].
$$

Здесь $A(r,t)$ — mass/activity field, а color cluster labels фиксируются глобально по всей simulation, не отдельно по кадрам. Далее считается change signal

$$
G_c(t)=|M_c(t+\delta)-M_c(t)|.
$$

Корреляция:

$$
\rho_c=\operatorname{corr}\left(\Delta H(t),G_c(t)\right).
$$

Этот анализ должен быть подписан как validated-color case study. Он не переносится автоматически на MSPD-optimized runs, где цвет может быть несемантическим.

### Что рисовать

Один timeline plot: $\Delta H(t)$, несколько $M_c(t)$ или $G_c(t)$, вертикальные маркеры визуально аннотированных transitions. Под ним — 4–6 frames до/во время/после major peaks. В caption явно написать: color analysis is used only in a simulation where color-role alignment was manually validated.

## Эксперимент C2-B. Branching sensitivity at high-$\Delta H$ times

### Зачем нужен эксперимент

Это основной proof для C2. Он не требует species labels, object tracking, patch motifs или цветовой semantics. Он проверяет causal/dynamical meaning $\Delta H$: если $\Delta H$ spike действительно указывает на transition-sensitive state, то малые perturbations в такие моменты должны сильнее менять будущее.

### Что будет, если его не провести

Reviewer может сказать, что $\Delta H$ spike — это просто резкое движение большой массы, rendering artifact или временное увеличение скорости. Корреляция с визуальной сменой видов на одной симуляции не доказывает general mechanism.

### Что ожидается увидеть

Для high-$\Delta H$ моментов branching divergence должен быть выше, чем для matched low-$\Delta H$ моментов:

$$
\mathbb E[B(t)\mid t\in T_{\mathrm{high}}]>
\mathbb E[B(t)\mid t\in T_{\mathrm{low}}].
$$

Если эффект есть на flagship simulation и хотя бы на нескольких MSPD-opt simulations, claim становится сильным. Если эффект есть только на flagship, C2 надо позиционировать как mechanistic case study plus calibrated diagnostic.

### Подробный protocol

Берём одну или несколько simulations, для которых уже есть $\Delta H(t)$. Основной набор: flagship complex simulation и top $2$–$3$ MSPD-opt simulations по $D_{\mathrm{eval}}$. Это ограничивает compute.

Сначала строится smoothed time series:

$$
\bar{H}(t)=\operatorname{Smooth}\left(\Delta H(W_t,\tau^\star)\right).
$$

Выбираются high times:

$$
T_{\mathrm{high}}=\{t_1,\dots,t_m\},
$$

где $t_i$ — локальные максимумы $\bar H(t)$ в top quantile. Чтобы не брать соседние точки одного события, вводится refractory interval $r_H$: два выбранных $t_i$ должны быть разделены минимум на $r_H$ steps.

Выбираются low/mid matched times:

$$
T_{\mathrm{low}}=\{t'_1,
\dots,t'_m\},
$$

так чтобы $\bar H(t'_i)$ лежал в нижнем или среднем quantile. По возможности matching делается по total mass/activity:

$$
A_{\mathrm{tot}}(t)=\sum_r A(r,t),
$$

и mean speed/activity:

$$
V_{\mathrm{mean}}(t)=\mathbb E_i\|u_i^{(\tau_0)}(t)\|.
$$

То есть high и low times не должны отличаться только тем, что в high time система просто более активна. Matching можно сделать грубо: выбирать low times из того же bin по $A_{\mathrm{tot}}$ и $V_{\mathrm{mean}}$.

В каждый выбранный момент $t$ создаётся $R$ branches:

$$
X_t^{(b)}=X_t+\epsilon\eta_b,
\qquad b=1,
\dots,R,
$$

где $\eta_b$ — independent small perturbation. Perturbation должна быть достаточно малой, чтобы не менять состояние макроскопически. Например, small Gaussian noise в $A/P/F$, или small displacement/noise в particle coordinates, в зависимости от substrate.

Каждая branch симулируется на horizon $T_B$. Получаются continuations:

$$
X_{t:t+T_B}^{(1)},
\dots,
X_{t:t+T_B}^{(R)}.
$$

Future divergence считается по embedding clouds или по trajectory descriptors. Для rendered-frame/CLIP версии:

$$
\mathcal Z^{(b)}_{t}=
\{z(X^{(b)}(t+\ell)): \ell\in L_B\},
$$

где $z(\cdot)$ — CLIP/OpenNNS frame embedding, а $L_B$ — sampled future frames.

Расстояние между branches:

$$
d_{\mathrm{emb}}(a,b)=\operatorname{Chamfer}\left(\mathcal Z^{(a)}_t,
\mathcal Z^{(b)}_t\right).
$$

Branching score:

$$
B_{\mathrm{emb}}(t)=
\frac{2}{R(R-1)}
\sum_{a<b}d_{\mathrm{emb}}(a,b).
$$

Если trajectory data дешёво доступна, дополнительно считается trajectory branching score:

$$
B_{\mathrm{traj}}(t)=
\frac{2}{R(R-1)}
\sum_{a<b}
\left|D_{\mathrm{eval}}(X^{(a)}_{t:t+T_B})-D_{\mathrm{eval}}(X^{(b)}_{t:t+T_B})\right|
$$

или distance между $\Delta H(W,\tau)$ maps двух branches.

### Что рисовать

Основной plot: paired high vs low branching scores. Для каждого matched pair $(t_i,t'_i)$ провести линию между $B(t'_i)$ и $B(t_i)$. Отдельно показать median difference и bootstrap CI.

Вторая панель: scatter $\bar H(t)$ vs $B(t)$ по выбранным high/low times.

Третья панель: 2–3 visual branch examples: состояние в момент $t$, затем несколько future frames по branches. Это делает causal interpretation понятной.

---

# 3. Claim C3: temporal scale separation and $\tau$

## Исходная гипотеза

MSPD reveals scale separation at system-identified $\tau^\star$; velocity distributions split into slow glider-core and fast medium/periphery clusters.

## Предлагаемая формулировка

In high-complexity simulations, MSPD-induced trajectory geometry can expose mesoscopic temporal scales at which dynamic roles become separable, stable, and spatially organized. The lag selected by the MSPD objective should be compared to an independently measured role-separation lag, rather than assumed to be the same object by definition.

Эта формулировка важна из-за оптимизируемого $\tau$. Нельзя одновременно сказать, что $\tau$ выбран objective-ом и что он независимо найден clustering-ом, если это не проверено. Поэтому вводятся два лага:

$$
\tau_{\mathrm{MSPD}}=\tau^\star(\theta)
$$

и

$$
\tau_{\mathrm{role}}=
\arg\max_{\tau\in\mathcal T}Q(\tau),
$$

где $Q(\tau)$ — independent role-separation score.

## Эксперимент C3. Independent role-separation profile

### Зачем нужен эксперимент

Он проверяет, имеет ли selected temporal scale физический смысл. Если $\tau_{\mathrm{MSPD}}$ близок к $\tau_{\mathrm{role}}$, то MSPD не просто максимизирует абстрактную heterogeneity, а выбирает scale, на котором trajectory roles становятся separable.

### Что будет, если его не провести

C3 останется hand-picked case study. Reviewer может сказать, что $\tau$ просто optimized nuisance parameter, а не discovered temporal scale.

### Что ожидается увидеть

На flagship simulation желательно:

$$
\tau_{\mathrm{MSPD}}
\approx
\tau_{\mathrm{role}}
$$

или хотя бы оба лага должны лежать в одном high-signal plateau. Кластеры при $\tau_{\mathrm{role}}$ должны иметь разные physical statistics: например, slow/core-like vs fast/periphery-like. Они также должны быть spatially coherent, а не случайно раскиданы по системе.

На всех 9 optimized runs это может не реплицироваться. Тогда claim формулируется как mechanistic case study: “in a representative high-complexity ecology”. Population-level claim делается только если совпадение $\tau_{\mathrm{MSPD}}$ и $\tau_{\mathrm{role}}$ наблюдается на нескольких independent runs.

### Подробный protocol

Для выбранной simulation и каждого $\tau\in\mathcal T$ строятся эмпирические распределения скоростей частиц/tracers:

$$
p_i^{(\tau)}=
\frac{1}{|T|}
\sum_{t\in T}\delta_{u_i^{(\tau)}(t)}.
$$

Попарная distance matrix:

$$
R_{ij}^{(\tau)}=
\operatorname{SW}_1(p_i^{(\tau)},p_j^{(\tau)}).
$$

Дальше по $R^{(\tau)}$ делается clustering. Для основного claim про core/periphery достаточно фиксировать $K=2$, чтобы не вводить ещё одну степень свободы. Если $K=2$ плохо описывает flagship simulation, можно в supplementary показать $K=3,4$, но primary plot должен быть без подбора $K$.

Labels:

$$
\ell_i^{(\tau)}\in\{1,2\}.
$$

Separability:

$$
S(\tau)=\operatorname{silhouette}\left(R^{(\tau)},\ell^{(\tau)}\right).
$$

Bootstrap stability: bootstrap-им time chunks или windows, пересчитываем $R^{(\tau,b)}$, clustering и labels $\ell^{(\tau,b)}$. Stability:

$$
\operatorname{Stab}(\tau)=
\mathbb E_{b,b'}
\operatorname{ARI}\left(
\ell^{(\tau,b)},
\ell^{(\tau,b')}
\right),
$$

где ARI — adjusted Rand index.

Spatial coherence: в каждом reference frame или averaged over frames строится spatial kNN graph по координатам частиц/tracers. Тогда

$$
C_{\mathrm{sp}}(\tau)=
\Pr(\ell_i^{(\tau)}=\ell_j^{(\tau)}\mid j\in \operatorname{kNN}(i))-
\Pr(\ell_i^{(\tau)}=\ell_j^{(\tau)}\mid i,j \text{ random}).
$$

Functional separation: для каждого cluster-а считаются speed/displacement statistics:

$$
V_k(\tau)=
\mathbb E_{i:\ell_i^{(\tau)}=k}
\mathbb E_t\|u_i^{(\tau)}(t)\|.
$$

И path efficiency:

$$
P_k(\tau)=
\mathbb E_{i:\ell_i^{(\tau)}=k}
\frac{\|x_i(t_2)-x_i(t_1)\|}
{\sum_{t=t_1}^{t_2-1}\|x_i(t+1)-x_i(t)\|+\epsilon}.
$$

Independent role score:

$$
Q(\tau)=
z(S(\tau))+z(\operatorname{Stab}(\tau))+z(C_{\mathrm{sp}}(\tau)),
$$

where $z(\cdot)$ means standardization across $\tau\in\mathcal T$. Then

$$
\tau_{\mathrm{role}}=\arg\max_\tau Q(\tau).
$$

This $Q$ is not claimed as a new universal complexity metric. It is only an independent diagnostic for whether the MSPD-selected lag corresponds to separable dynamical roles.

### Что рисовать

Figure C3:

1. Curves $D(\theta,\tau)$, $S(\tau)$, $\operatorname{Stab}(\tau)$, $C_{\mathrm{sp}}(\tau)$, and $Q(\tau)$ over $\log \tau$. Vertical lines: $\tau_{\mathrm{MSPD}}$ and $\tau_{\mathrm{role}}$.
2. Distance matrices $R^{(\tau)}$ at small $\tau$, $\tau_{\mathrm{role}}$, large $\tau$.
3. Spatial overlay of cluster labels at $\tau_{\mathrm{role}}$ on representative frame.
4. Speed/path distributions by cluster.

---

# 4. Claim C4: ecological richness and non-degenerate organization — deferred

## Исходная гипотеза

MSPD-optimized worlds match NN-OEE baselines in ecological richness despite using an explicit three-line formula.

## Текущий статус

Этот claim пока лучше не включать в основной experimental plan. Причина не в том, что claim неверный, а в том, что сейчас нет надёжного operational definition для real biodiversity/ecological richness в dense Flow-Lenia.

Object tracking, connected components, SAM/cell-detectors, color-based species counts и handcrafted patch motifs не являются приемлемой main evidence: объекты слипаются, масштаб меняется, цвет может быть несемантическим, активный субстрат может образовывать одну большую связную компоненту, а локальные patch features не дают устойчивой species/morphology ontology.

Handcrafted descriptors вроде embedding diversity, trajectory persistence, coherent transport или velocity heterogeneity тоже не должны использоваться как main proof of biodiversity. Они могут быть supplementary sanity diagnostics, но они легко интерпретируются как activity/transport/entropy, а не как ecological richness.

## Предлагаемая будущая формулировка

Если C4 нужно будет вернуть, я бы формулировал его так:

> MSPD-optimized simulations are preferentially judged to exhibit richer ecological organization than matched random controls under a calibrated semantic phenotype assay. This is interpreted as perceived ecological organization, not as ground-truth taxonomic biodiversity.

Для этого наиболее реалистичный вариант без human assessor markup — VLM/LLM-like pairwise evaluation по APF-derived snapshots или videos, preferably с calibration на synthetic regimes и swapped order для контроля position bias.

## Почему C4 скипается сейчас

Если оставить C4 в текущем виде, основной риск такой: reviewer спросит, что именно считается biodiversity. Любая простая метрика будет выглядеть как surrogate, который можно захакать и который не измеряет species richness. Поэтому в текущей версии experimental plan C4 не используется как основной claim и не требует новых экспериментов.

## Optional future experiment: calibrated semantic phenotype assay

Этот блок можно добавить позже, если будет время.

Данные: matched pairs из уже имеющихся Flow-Lenia groups. Для каждой группы берётся MSPD-optimized clip и matched random clip. Если есть NN-OEE exemplars, они используются только как reference, а не как powered statistical group.

Judge получает два anonymized candidates A/B и выбирает, где richer ecological organization. Prompt фиксирует критерии: persistent coherent agents, multiple morphologies/roles, interactions, split/merge/absorption-like events, non-degenerate temporal persistence. Не просить оценивать beauty, colorfulness или rendering style.

Каждая пара прогоняется в двух orders: A/B и B/A. Inconsistent swapped decisions учитываются как tie. Target statistic:

$$
w_p=I[\text{MSPD chosen}]+\frac{1}{2}I[\text{tie}].
$$

Для group-level aggregation:

$$
W_r=\frac{1}{3L}\sum_{j=1}^{3}\sum_{\ell=1}^{L}w_{r,j,\ell}.
$$

Primary test:

$$
H_0:\operatorname{median}(W_r)=0.5,\quad H_1:\operatorname{median}(W_r)>0.5.
$$

Calibration pairs из synthetic suite должны иметь известный expected ordering, например structured multi-role dynamics preferred over homogeneous Brownian or static dynamics. Calibration accuracy and swapped-order consistency should be reported before interpreting target Flow-Lenia results.

В текущем плане этот experiment не является обязательным.

---

# 5. Claim C5: frustration

## Исходная гипотеза

Frustration is structural/snapshot, not dynamic/MSPD. NN snapshot metric shows frustration in most runs, MSPD axis only in fewer runs. Snapshot and dynamic complexity are distinct axes.

## Предлагаемая формулировка

MSPD-optimized systems exhibit increased blockwise frustration: suppressing spatial coupling during early development produces larger structural divergence than ordinary seed stochasticity. The frustration signal is strongest on snapshot/representation axes and may be weaker on raw MSPD dynamic axes, indicating that structural and trajectory-distributional complexity are empirically dissociable.

Это не утверждает абсолютную orthogonality. Оно утверждает measurable dissociation.

## Эксперимент C5. Matched A/B/C blockwise frustration assay

### Зачем нужен эксперимент

Это основной proof, что MSPD optimization создаёт integrated systems, где independent block development несовместим с global development. Это ближе к физическому понятию frustration, чем просто “complexity score high”.

### Что будет, если его не провести

Тогда статья показывает только рост MSPD, но не показывает Vanchurin-style frustration or integration. Claim про frustration останется unsupported.

### Что ожидается увидеть

Для каждой system $\theta$:

$$
F_q(\theta)=d_q(A_\theta,B_\theta)-d_q(A_\theta,C_\theta).
$$

Здесь:

- $A_\theta$: normal simulation with seed $s_0$;
- $B_\theta$: same seed $s_0$, first 500k steps with $2\times2$ isolation walls, then walls removed, then measured after washout;
- $C_\theta$: normal simulation with different seed $s_1$;
- $d_q$: distance between the measured final segments under metric axis $q$.

Frustration exists if

$$
F_q(\theta)>0.
$$

MSPD optimization increases frustration if

$$
\Delta_r^{C5,q}=F_q(\theta_r^{\mathrm{opt}})-
\operatorname{median}_{j=1}^3F_q(\theta_{r,j}^{\mathrm{rand}})>0
$$

for most groups $r$.

### Подробный protocol

For every checkpoint $\theta$, use the three already defined simulations.

A:

$$
X_{\theta,s_0}^{A}(0:1.2M)
$$

normal simulation.

B:

$$
X_{\theta,s_0}^{B}(0:500k)
$$

with $2\times2$ isolated blocks; then

$$
X_{\theta,s_0}^{B}(500k:1.0M)
$$

normal global dynamics after removing walls; then

$$
X_{\theta,s_0}^{B}(1.0M:1.2M)
$$

measurement segment.

C:

$$
X_{\theta,s_1}^{C}(0:1.2M)
$$

normal simulation with different seed.

For snapshot/representation axis, define frame embedding clouds over the final segment:

$$
\mathcal Z_A(\theta)=\{z(I_t^A):t\in T_{\mathrm{meas}}\},
$$

and similarly $\mathcal Z_B,\mathcal Z_C$. Chamfer distance:

$$
\operatorname{Ch}(\mathcal Z_1,\mathcal Z_2)=
\frac{1}{2}
\left[
\frac{1}{|\mathcal Z_1|}\sum_{z\in\mathcal Z_1}
\min_{z'\in\mathcal Z_2}\|z-z'\|_2
+
\frac{1}{|\mathcal Z_2|}\sum_{z'\in\mathcal Z_2}
\min_{z\in\mathcal Z_1}\|z'-z\|_2
\right].
$$

Then

$$
F_{\mathrm{emb}}(\theta)=
\operatorname{Ch}(\mathcal Z_A,\mathcal Z_B)-
\operatorname{Ch}(\mathcal Z_A,\mathcal Z_C).
$$

For MSPD/dynamic axis, compute $\Delta H(W,\tau)$ maps for A/B/C using each system's selected $\tau^\star$, or compute full $\tau$-profiles and compare maps:

$$
\mathcal M_A(\theta)=\{\Delta H_A(W,\tau):W\in\mathcal W,\tau\in\mathcal T\}.
$$

Dynamic distance can be

$$
d_{\mathrm{MSPD}}(A,B)=
\left\|\mathcal M_A-\mathcal M_B\right\|_1
$$

after matching window indices and normalizing each map by its median absolute deviation. Then

$$
F_{\mathrm{MSPD}}(\theta)=d_{\mathrm{MSPD}}(A,B)-d_{\mathrm{MSPD}}(A,C).
$$

If current implementation already defines a dynamic frustration distance, use that implementation and write its exact formula.

### Что рисовать

Figure C5:

1. Protocol diagram A/B/C.
2. Per-checkpoint $d(A,B)$ vs $d(A,C)$ scatter. Points above diagonal have positive frustration.
3. Matched group contrast $\Delta_r^{C5,\mathrm{emb}}$: optimized minus median random.
4. Same for $\Delta_r^{C5,\mathrm{MSPD}}$. This shows whether frustration is stronger on snapshot axis than dynamic axis.
5. Axis dissociation plot: $F_{\mathrm{emb}}$ vs $F_{\mathrm{MSPD}}$. Do not call it orthogonality unless correlation is actually low and confidence interval supports it.

### Optional cheap extension: washout curve

If frames/states are saved for multiple post-release intervals, compute

$$
F_q(u)=d_q(A_{u:u+w},B_{u:u+w})-d_q(A_{u:u+w},C_{u:u+w}),
$$

where $u$ is time after wall removal. If $F_q(u)$ persists after washout, frustration is less likely to be a release shock. This is useful but not required if it needs expensive reruns.

---

# 6. Claim C6: transfer beyond Flow-Lenia

## Исходная гипотеза

MSPD generalizes beyond Flow-Lenia to Boids and Particle Life.

## Предлагаемая формулировка

The MSPD protocol transfers to independent particle substrates. In Particle Life++, MSPD optimization increases trajectory-distributional complexity and blockwise frustration under the same analysis pipeline. Lower-capacity substrates such as Boids can be reported as limited-expressivity controls rather than full ecological demonstrations.

## Эксперимент C6. Same pipeline on Particle Life++

### Зачем нужен эксперимент

Он показывает, что MSPD не является Flow-Lenia-specific artifact. Особенно важно, что Particle Life++ имеет другую mechanics and parameterization.

### Что будет, если его не провести

Статья будет по сути Flow-Lenia paper. Claim про generality будет unsupported.

### Что ожидается увидеть

For Particle Life++:

$$
D_{\mathrm{eval}}(\theta^{\mathrm{opt}})>
D_{\mathrm{eval}}(\theta^{\mathrm{rand}})
$$

and

$$
F_{\mathrm{emb}}(\theta^{\mathrm{opt}})>
F_{\mathrm{emb}}(\theta^{\mathrm{rand}})
$$

in matched groups. Если есть 6/7 positive groups, это нормальное evidence. Если Particle Life++ требует activation regularizer, это надо не скрывать, а описать как escape from zero-motion basin.

### Подробный protocol

Использовать тот же C1/C5 pipeline:

1. For each Particle Life++ group $r$, one optimized checkpoint and three random controls.
2. For each checkpoint, simulations A/B/C if available.
3. Compute $\tau^\star(\theta)$ by same selection-adjusted rule.
4. Compute $D_{\mathrm{eval}}(\theta)$.
5. Compute $F_{\mathrm{emb}}(\theta)$ and, if available, $F_{\mathrm{MSPD}}(\theta)$.
6. Do not compute C4-specific organization descriptors in the main pipeline; if C4 is revived later, reuse saved APF snapshots/videos for semantic phenotype evaluation.

For cross-substrate comparison, report normalized effect sizes:

$$
Z_D^{(S)}=
\frac{
\operatorname{median}_{r}D_{\mathrm{eval}}(\theta_r^{\mathrm{opt}})-
\operatorname{median}_{r,j}D_{\mathrm{eval}}(\theta_{r,j}^{\mathrm{rand}})
}{
\operatorname{MAD}_{r,j}D_{\mathrm{eval}}(\theta_{r,j}^{\mathrm{rand}})+\epsilon
},
$$

where $S$ denotes substrate: Flow-Lenia or Particle Life++.

Analogously:

$$
Z_F^{(S)}=
\frac{
\operatorname{median}_{r}F_{\mathrm{emb}}(\theta_r^{\mathrm{opt}})-
\operatorname{median}_{r,j}F_{\mathrm{emb}}(\theta_{r,j}^{\mathrm{rand}})
}{
\operatorname{MAD}_{r,j}F_{\mathrm{emb}}(\theta_{r,j}^{\mathrm{rand}})+\epsilon
}.
$$

### Minimal regularizer check for Particle Life++

Because Particle Life++ starts in a zero-motion basin, activation regularization is not a minor implementation detail. But this does not require a large ablation grid.

Minimal check:

- one or two no-regularizer runs showing zero-motion/stagnation;
- one activation-only run showing that activation can create motion but not necessarily high MSPD/frustration;
- main MSPD+activation runs showing high $D_{\mathrm{eval}}$ and positive frustration.

This is not a hyperparameter robustness sweep. It only establishes that the final result is not trivially explained by the activation term alone.

### Что рисовать

Figure C6:

1. Flow-Lenia vs Particle Life++ normalized effect sizes $Z_D$ and $Z_F$.
2. Particle Life++ optimization curves: MSPD score, mean motion/activity, optional activation term.
3. Example frames/clips: random Particle Life++, optimized Particle Life++.
4. If Boids included: show it separately as limited-expressivity substrate, not as a failed ecology.

---

# 7. Synthetic calibration suite: canonical pattern calibration

Этот блок **нельзя выкидывать**. Это не набор точечных anti-jitter tests, а калибровка измерительного прибора на системах с известной ground truth. Он должен идти рано в Results: сначала показываем, что MSPD ведёт себя осмысленно на простых canonical dynamics, и только потом интерпретируем Flow-Lenia.

## Hypothesis tested

MSPD responds to heterogeneity of trajectory distributions, not to arbitrary visual complexity, global texture, or motion per se. In systems with known roles, known transition times, and known temporal scales, the estimator should recover the intended qualitative structure:

$$
\text{no trajectory heterogeneity} \Rightarrow D(\tau) \approx 0,
$$

$$
\text{coexisting dynamic roles} \Rightarrow D(\tau)>0,
$$

$$
\text{partial/staggered regime transition} \Rightarrow \Delta H(t) \text{ peaks during role coexistence},
$$

$$
\text{multi-scale roles} \Rightarrow \tau^\star \text{ or role-separation scale is interpretable}.
$$

This calibration also clarifies what MSPD should **not** detect. A visually noisy field with no coherent trajectory structure should not produce a high score. A synchronous global change of all particles should not necessarily create high MSPD, because MSPD is a heterogeneity measure, not a generic change-point detector.

## Зачем нужен эксперимент

Без этого блока все выводы про $\Delta H$, $\tau^\star$ и transition sensitivity проверяются только на Flow-Lenia, где нет ground-truth species, нет ground-truth events и нет формальной разметки ролей. Synthetic calibration даёт дешёвую независимую проверку того, что метрика измеряет именно заявленный класс явлений.

## Что будет, если его не провести

Reviewer сможет сказать, что MSPD является post-hoc interpretation of Flow-Lenia videos: мы видим красивую симуляцию, видим высокий score, и потом называем это complexity. Calibration suite превращает этот аргумент в проверяемый вопрос: на простых системах с известной структурой метрика ведёт себя предсказуемо.

## Общий protocol

Primary implementation should be **trajectory-level**, not video-level. Это важно: для калибровки не нужен APF tensor и не нужна Flow-Lenia. Мы напрямую генерируем synthetic trajectories

$$
x_i(t)\in [0,L)^2,
\qquad i=1,\dots,n,
\qquad t=0,\dots,T,
$$

на двумерном торе. Эти trajectories подаются в тот же MSPD estimator, что и реальные tracer trajectories. Все synthetic families используют одну и ту же сетку лагов $\mathcal T$, один и тот же windowing protocol и одну и ту же формулу $D(\theta,\tau)$.

Recommended default:

$$
L=1,
\qquad n\in\{256,512\},
\qquad T\in[10^4,10^5],
$$

with $3$--$5$ seeds per family. Это не оптимизация и не full Flow-Lenia rollout, поэтому compute должен быть маленьким.

For each family compute:

$$
D(\tau),
\qquad
\tau^\star=\arg\max_{\tau\in\mathcal T}D(\tau),
\qquad
\Delta H(W,\tau),
$$

and, when ground-truth labels are available,

$$
\operatorname{ARI}(\ell^{(\tau)},g),
$$

where $g_i$ is the known synthetic role label and $\ell_i^{(\tau)}$ is a clustering label obtained from the MSPD/SW distance matrix at lag $\tau$.

For event families with known transition time $t_0$, compute

$$
\hat t=\arg\max_t \Delta H(t),
\qquad
E_t=|\hat t-t_0|,
$$

or, if the transition lasts over an interval $[t_0,t_1]$,

$$
E_t=\operatorname{dist}(\hat t,[t_0,t_1]).
$$

Ground-truth labels are used only for evaluation, not for computing MSPD.

---

## Family S0. Static particles / static field

### Definition

Particles do not move:

$$
x_i(t+1)=x_i(t).
$$

Optional rendered-field version: a static scalar field $A(x)$ or static Gaussian texture is displayed, but tracer coordinates remain fixed.

### Ground truth

No trajectory dynamics and no trajectory-distributional heterogeneity.

### Expected result

$$
D(\tau)\approx 0,
\qquad
\Delta H(W,\tau)\approx 0.
$$

### Interpretation

If MSPD is high here, the implementation is contaminated by finite-sample artifacts, rendering artifacts, or baseline normalization errors.

---

## Family S1. Homogeneous Gaussian/Brownian motion

### Definition

All particles follow the same iid Gaussian velocity law:

$$
x_i(t+1)=x_i(t)+\sigma\xi_i(t),
\qquad
\xi_i(t)\sim\mathcal N(0,I_2),
$$

with periodic boundary conditions.

Equivalent velocity-field version:

$$
F_t(x)=\sigma\xi_t(x),
$$

where $\xi_t(x)$ is white or short-correlated Gaussian noise, and tracers are advected by $F_t$.

### Ground truth

Motion exists, but all particles have the same trajectory law. There are no persistent dynamic roles.

### Expected result

$$
D(\tau) \text{ is low for all } \tau,
\qquad
\operatorname{ARI} \text{ is not meaningful because no roles exist}.
$$

Finite-sample MSPD may be nonzero, but it should define the empirical null scale.

### Interpretation

This family calibrates the difference between “there is motion/noise” and “there is heterogeneous organization”. It directly answers the concern that MSPD might simply reward Gaussian stochasticity.

---

## Family S2. Visual Gaussian flicker with fixed trajectories

This family is optional if MSPD only consumes trajectories. It is useful if a reviewer might confuse rendered visual complexity with MSPD complexity.

### Definition

Generate a visually noisy field:

$$
A_t(x)=\mu+\sigma\zeta_t(x),
\qquad
\zeta_t(x)\sim\mathcal N(0,1),
$$

but keep tracer positions fixed:

$$
x_i(t+1)=x_i(t).
$$

### Ground truth

The video is visually active, but there is no trajectory dynamics.

### Expected result

$$
D(\tau)\approx 0.
$$

### Interpretation

This shows that MSPD is not a rendered-video texture metric. If this is too far from the current implementation, omit it from main text and keep S1 as the Gaussian null.

---

## Family S3. One coherent moving blob

### Definition

A single coherent object moves with velocity $v$. Generate a center

$$
c(t+1)=c(t)+v,
$$

and particles around the center:

$$
x_i(t)=c(t)+r R_i+\epsilon_i(t),
$$

where $R_i$ is a fixed vector sampled once from a compact distribution, for example inside the unit disk, and $\epsilon_i(t)$ is small within-blob noise.

Two variants are useful:

**S3-active-only:** all sampled particles belong to the blob.

**S3-blob-plus-background:** a fraction of particles belong to the moving blob and the rest are static or Brownian background.

### Ground truth

S3-active-only has coherent transport but no role diversity. S3-blob-plus-background has exactly two roles: moving blob and background.

### Expected result

For active-only:

$$
P_{\mathrm{path}} > 0,
\qquad
D(\tau) \text{ low or moderate},
$$

because all active particles share nearly the same trajectory law.

For blob-plus-background:

$$
D(\tau)>0,
\qquad
\operatorname{ARI}(\ell^{(\tau)},g) \text{ should be high},
$$

because the metric should separate moving and non-moving roles.

### Interpretation

This calibrates a crucial distinction: MSPD should not reward mere coherent motion as rich ecology. It should increase when distinct dynamical roles coexist.

---

## Family S4. Two-speed or two-direction role mixture

### Definition

Particles are assigned to two or more known groups:

$$
g_i\in\{1,2\},
$$

with group-specific velocity laws:

$$
x_i(t+1)=x_i(t)+v_{g_i}+\sigma_{g_i}\xi_i(t).
$$

For example:

$$
\|v_1\|\ll \|v_2\|,
$$

or

$$
v_1=(a,0),
\qquad
v_2=(0,a).
$$

### Ground truth

There are persistent dynamic roles with known labels.

### Expected result

$$
D(\tau)>0,
\qquad
S(\tau)>0,
\qquad
\operatorname{ARI}(\ell^{(\tau)},g) \text{ high at informative } \tau.
$$

### Interpretation

This is the cleanest positive control for C3. If the trajectory-Wasserstein geometry cannot recover this, then scale-separation claims in Flow-Lenia are not credible.

---

## Family S5. Synchronous global regime switch

### Definition

All particles change their velocity law at the same known time $t_0$:

$$
x_i(t+1)=x_i(t)+v^{(1)}+\sigma\xi_i(t),
\qquad t<t_0,
$$

$$
x_i(t+1)=x_i(t)+v^{(2)}+\sigma\xi_i(t),
\qquad t\ge t_0.
$$

All particles switch together.

### Ground truth

There is a global change point, but no coexistence of different dynamic roles during the switch.

### Expected result

MSPD need not spike strongly:

$$
\Delta H(t) \text{ may stay low if all particles remain statistically identical}.
$$

### Interpretation

This is important. It shows that MSPD is not a generic change-point detector. It measures heterogeneity across trajectory distributions. A global synchronous change can be visually and dynamically real, but not necessarily high-MSPD.

---

## Family S6. Partial or staggered regime switch

### Definition

Only a subset of particles switches at $t_0$, or different particles switch over an interval $[t_0,t_1]$:

$$
x_i(t+1)=x_i(t)+v_{g_i(t)}+\sigma\xi_i(t),
$$

where

$$
g_i(t)\in\{\mathrm{old},\mathrm{new}\}.
$$

For a staggered transition, define switching times

$$
T_i\sim\operatorname{Uniform}(t_0,t_1),
$$

and

$$
g_i(t)=
\begin{cases}
\mathrm{old}, & t<T_i,\\
\mathrm{new}, & t\ge T_i.
\end{cases}
$$

### Ground truth

During $[t_0,t_1]$, old and new dynamic roles coexist. This is the synthetic analogue of an ecological turnover period.

### Expected result

$$
\Delta H(t) \text{ peaks inside or near } [t_0,t_1],
$$

$$
E_t=\operatorname{dist}(\hat t,[t_0,t_1]) \text{ is small},
$$

and clustering against true roles should improve during the coexistence interval.

### Interpretation

This is the cleanest calibration for C2. It makes precise why $\Delta H$ spikes during ecological transitions: not because “a species changed” abstractly, but because old and new dynamical roles coexist and create trajectory-distributional heterogeneity.

---

## Family S7. Multi-scale moving blobs

### Definition

Generate several coherent groups with different spatial sizes and speeds:

$$
x_i(t)=c_{g_i}(t)+r_{g_i}R_i+\epsilon_i(t),
$$

$$
c_g(t+1)=c_g(t)+v_g.
$$

Choose, for example, one large slow blob and several small fast blobs:

$$
r_1 \gg r_2,
\qquad
\|v_1\|\ll \|v_2\|.
$$

A natural characteristic time for group $g$ is

$$
\tau_g^{\mathrm{cross}}=\frac{r_g}{\|v_g\|+\epsilon}.
$$

### Ground truth

There are dynamic roles at different spatial and temporal scales.

### Expected result

The $D(\tau)$ and $Q(\tau)$ profiles should show a meaningful peak or plateau near scales where displacements distinguish the roles:

$$
\tau^\star \in \text{informative scale range determined by } \{\tau_g^{\mathrm{cross}}\}.
$$

Exact equality is not expected; the target is scale-range recovery, not point estimation.

### Interpretation

This is the synthetic version of the C3 claim. It tests whether optimized/selected $\tau$ can be interpreted as a mesoscopic timescale rather than a free hyperparameter.

---

## Family S8. Toy merge/split ecology

This one is optional; include only if it is easy to generate.

### Definition

Two blobs move independently, merge for a finite interval, and then split again:

$$
c_1(t),c_2(t) \rightarrow c_{12}(t) \rightarrow c_1'(t),c_2'(t).
$$

Particles keep latent labels, but during the merge their positions overlap and their velocities can be mixed.

### Ground truth

There is an interaction-like event with known timing, but no need for object tracking.

### Expected result

$\Delta H(t)$ or interaction-sensitive trajectory heterogeneity should increase near the merge/split intervals, and role separability may temporarily decrease during full overlap.

### Interpretation

This family checks whether MSPD responds sensibly to interaction-like events. It is useful, but less essential than S1, S3, S4, S6, and S7.

---

## Minimal synthetic suite

To keep compute small, the main paper does not need all families. The minimal set I would run is:

| Family | Why it is needed |
|---|---|
| S1 homogeneous Gaussian/Brownian motion | null for stochastic motion |
| S3 one coherent moving blob | motion without ecology |
| S4 two-role mixture | positive control for role heterogeneity |
| S5 synchronous global switch | shows MSPD is not generic change-point detection |
| S6 partial/staggered switch | positive control for transition-period $\Delta H$ |
| S7 multi-scale moving blobs | positive control for temporal-scale interpretation |

S0 static particles is so cheap that it should also be included as a sanity check. S2 visual flicker and S8 merge/split can be supplementary or omitted.

## Metrics computed on synthetic suite

For every synthetic sequence compute the MSPD observables:

$$
D(\tau),\qquad \Delta H(t),\qquad \tau^\star.
$$

Auxiliary physical descriptors may be shown only when they clarify a synthetic family, for example coherent transport in the one-blob case. They are not used as biodiversity or C4 metrics.

For S4, S6, S7 compute role agreement:

$$
\operatorname{ARI}(\ell^{(\tau)},g).
$$

For S5/S6/S8 compute event localization:

$$
\hat t=\arg\max_t\Delta H(t),
\qquad
E_t=\operatorname{dist}(\hat t,\text{ground-truth event interval}).
$$

For S7 compare selected scale to the known scale range:

$$
\tau^\star \in
[\min_g \tau_g^{\mathrm{cross}},\max_g \tau_g^{\mathrm{cross}}]
$$

or report distance to this range on a log scale.

## What to plot

Synthetic calibration figure should be compact and diagnostic.

Rows: S0/S1, S3, S4, S5, S6, S7.

Columns:

1. trajectory schematic or rendered toy snapshot;
2. $D(\tau)$ profile;
3. $\Delta H(t)$ with ground-truth event interval, when applicable;
4. role recovery or scale recovery, when applicable;
5. one-line interpretation.

The key visual message:

- Gaussian/Brownian motion: motion exists, MSPD low.
- One coherent blob: coherent transport exists, MSPD does not pretend this is rich ecology.
- Two-role mixture: MSPD detects coexistence of distinct trajectory laws.
- Synchronous switch: global change alone is not enough.
- Partial/staggered switch: $\Delta H$ peaks when old/new roles coexist.
- Multi-scale blobs: $\tau$ selection has an interpretable scale.

This single figure supports C1, C2 and C3 by showing how MSPD behaves on canonical regimes with known ground truth.

# 8. Integrated compute-aware execution plan

This section describes what to run and what to compute, minimizing duplicated simulations.

## Existing data block E1: Flow-Lenia A/B/C simulations

### Data

Use existing matched groups:

$$
\mathcal G_{FL}=\{\theta_r^{\mathrm{opt}},\theta_{r,1}^{\mathrm{rand}},\theta_{r,2}^{\mathrm{rand}},\theta_{r,3}^{\mathrm{rand}}\}_{r=1}^{G_{FL}}.
$$

For each checkpoint use simulations A, B, C as already defined.

### Compute from E1

For each checkpoint and each simulation A/C:

1. $\Delta H(W,\tau)$ for all $W\in\mathcal W$ and $\tau\in\mathcal T$.
2. $\tau^\star(\theta)$ via selection split.
3. $D_{\mathrm{eval}}(\theta)$.
4. Embedding clouds $\mathcal Z_A,\mathcal Z_B,\mathcal Z_C$ from final 200k steps.
5. Frustration scores $F_{\mathrm{emb}}(\theta)$ and $F_{\mathrm{MSPD}}(\theta)$ if dynamic map distance is available.

C4-specific organization descriptors are not part of the current main plan. If C4 is revived later, reuse the same clips/snapshots for semantic phenotype evaluation rather than adding a separate simulation batch.

### Claims covered

E1 covers C1 and C5. It also provides candidate simulations for C2 branching and C3 scale-separation case studies.

### Figures generated from E1

1. C1 profile plot and optimized-vs-random paired contrast.
2. C5 frustration scatter and matched contrast.
3. Representative heatmaps $\Delta H(W,\tau)$.

## Existing data block E2: NN-OEE exemplars

### Data

Use available NN-OEE optimized simulations. Do not present them as statistical group unless enough independent optimizations exist.

### Compute from E2

Compute the same MSPD observables as for Flow-Lenia normal simulation A:

$$
D(\tau),\quad \tau^\star,\quad D_{\mathrm{eval}}.
$$

If A/B/C NN-OEE simulations exist, compute frustration too; otherwise do not compare frustration statistically.

### Claims covered

E2 can contextualize C1 if NN-OEE has high MSPD, but should not be used for p-values. For C4 it remains only optional future reference material for semantic phenotype evaluation.

### Figures generated from E2

1. NN-OEE $D(\tau)$ profile as reference curve.
2. Representative frames/clips, if C4 is revived later.

## New cheap block N1: Synthetic canonical-pattern calibration

### Simulations to generate

Generate the minimal synthetic suite:

$$
\{S0,S1,S3,S4,S5,S6,S7\}.
$$

S2 visual Gaussian flicker and S8 merge/split are optional supplementary families. The suite is trajectory-level: no Flow-Lenia rollout and no APF tensor are required. Suggested scale:

$$
n\in\{256,512\},
\qquad
T\in[10^4,10^5],
\qquad
3\text{--}5 \text{ seeds per family}.
$$

Use exactly the same MSPD estimator, windowing, $\tau$ grid and selection rule as in the Flow-Lenia analysis.

### Compute from N1

For each synthetic sequence compute:

$$
D(\tau),\qquad \Delta H(t),\qquad \tau^\star.
$$

Optional auxiliary physical descriptors such as coherent transport or velocity persistence can be computed for interpretation, but they are not used as C4 evidence.

For S4/S6/S7 compute ARI against true role labels:

$$
\operatorname{ARI}(\ell^{(\tau)},g).
$$

For S5/S6 compute event-time localization error:

$$
E_t=\operatorname{dist}(\arg\max_t\Delta H(t),\text{ground-truth event interval}).
$$

For S7 compare selected $\tau^\star$ to the known synthetic scale range.

### Claims covered

N1 supports C1, C2 and C3 by calibrating the instrument. It does not replace Flow-Lenia experiments; it makes their interpretation credible. It is not a pointwise artifact checklist. It is the main evidence that MSPD distinguishes homogeneous stochastic motion, coherent but simple motion, role heterogeneity, transition-period heterogeneity and multi-scale dynamics.

### Figures generated from N1

Synthetic canonical-pattern calibration grid.

## New moderate block N2: Branching sensitivity

### Simulations to run

Select:

- one flagship complex simulation;
- top 2–3 MSPD-opt Flow-Lenia simulations by $D_{\mathrm{eval}}$;
- optionally one random control simulation.

For each selected simulation choose $m$ high-$\Delta H$ times and $m$ matched low/mid-$\Delta H$ times. Suggested first pass: $m=5$ high and $m=5$ low per simulation, $R=4$ branches per time, horizon $T_B$ chosen to be long enough for visible divergence but much shorter than full 1.2M runs.

Total branch count per simulation:

$$
2mR.
$$

With $m=5,R=4$, this is 40 short branches per selected simulation.

### Compute from N2

For each selected time $t$:

$$
B_{\mathrm{emb}}(t)=
\frac{2}{R(R-1)}
\sum_{a<b}
\operatorname{Ch}(\mathcal Z_t^{(a)},\mathcal Z_t^{(b)}).
$$

Optionally compute dynamic branch divergence from $\Delta H$ maps if cheap.

### Claims covered

N2 is the main strong proof for C2.

### Figures generated from N2

1. High vs low branching paired plot.
2. $\Delta H(t)$ vs branching divergence scatter.
3. Visual branch examples.

## Offline block N3: Scale-separation analysis

### Data used

Use already saved trajectories from flagship simulation and optionally top 2–3 MSPD-opt simulations. No new simulation should be required if trajectories are available.

### Compute from N3

For each $\tau\in\mathcal T$:

$$
R^{(\tau)},\quad S(\tau),\quad \operatorname{Stab}(\tau),
\quad C_{\mathrm{sp}}(\tau),
\quad Q(\tau),
\quad \tau_{\mathrm{role}}.
$$

Compare to $\tau_{\mathrm{MSPD}}$.

### Claims covered

N3 supports C3.

### Figures generated from N3

Scale-separation figure with profiles, distance matrices, spatial overlay, and role statistics.

## Existing/new block E3/N4: Particle Life++ transfer

### Data

Use existing Particle Life++ matched groups if available. If activation regularizer is involved, collect minimal baseline evidence rather than full ablation grid.

### Compute from E3

Same as Flow-Lenia E1 where applicable:

$$
D_{\mathrm{eval}},\quad F_{\mathrm{emb}}.
$$

### Minimal new runs if needed

Run one or two no-regularizer or activation-only baselines only to show that activation alone does not explain final MSPD/frustration. Do not run a large regularizer hyperparameter sweep.

### Claims covered

E3/N4 supports C6.

### Figures generated

Cross-substrate effect-size plot and Particle Life++ examples.

---

# 9. Final figure plan

## Figure 1. Synthetic calibration of MSPD

Purpose: establish what MSPD detects in controlled systems.

Panels:

- rows S0/S1/S3/S4/S5/S6/S7;
- trajectory schematic;
- $D(\tau)$;
- $\Delta H(t)$ with ground-truth event marker where applicable;
- $Q(\tau)$ or ARI for known role mixtures;
- short interpretation.

Claims supported: C1, C2, C3.

## Figure 2. Flow-Lenia MSPD optimization increases trajectory-distributional complexity

Purpose: C1 main result.

Panels:

- $D(\theta,\tau)$ profiles for random and MSPD-opt;
- paired optimized-vs-random $D_{\mathrm{eval}}$ contrasts;
- representative $\Delta H(W,\tau)$ heatmaps.

## Figure 3. $\Delta H$ peaks mark transition-sensitive states

Purpose: C2.

Panels:

- flagship timeline $\Delta H(t)$ plus annotated ecological transitions;
- high vs low branching divergence;
- $\Delta H(t)$ vs branching score scatter;
- branch visual examples.

## Figure 4. MSPD temporal scale and role separation

Purpose: C3.

Panels:

- $D(\tau)$, $S(\tau)$, $\operatorname{Stab}(\tau)$, $C_{\mathrm{sp}}(\tau)$, $Q(\tau)$;
- vertical lines for $\tau_{\mathrm{MSPD}}$ and $\tau_{\mathrm{role}}$;
- distance matrices;
- spatial overlay;
- speed/path distributions by cluster.

## Figure 5. Blockwise frustration in MSPD-optimized systems

Purpose: C5.

Panels:

- A/B/C protocol;
- $d(A,B)$ vs $d(A,C)$ scatter;
- matched optimized-vs-random $F_{\mathrm{emb}}$ contrast;
- matched $F_{\mathrm{MSPD}}$ contrast;
- $F_{\mathrm{emb}}$ vs $F_{\mathrm{MSPD}}$ dissociation plot.

## Figure 6. Transfer to Particle Life++

Purpose: C6.

Panels:

- normalized effect sizes $Z_D$, $Z_F$ for Flow-Lenia and Particle Life++;
- Particle Life++ optimization curves;
- representative random vs optimized Particle Life++ frames/clips;
- optional activation-only/no-regularizer minimal control.

## Deferred optional figure. Semantic phenotype assay for C4

Only add this if C4 is revived. It should use calibrated pairwise VLM/LLM-like evaluation on APF snapshots or videos, with swapped order and synthetic calibration pairs. It should not be presented as ground-truth biodiversity.

---

# 10. Recommended priority order

The most compute-efficient order is:

1. Compute all E1 metrics from existing Flow-Lenia A/B/C simulations. This immediately gives C1, C5 and candidate cases for C2/C3.
2. Generate synthetic calibration suite N1. This is cheap and strengthens interpretation across claims.
3. Run branching sensitivity N2 on flagship plus 2–3 top MSPD-opt simulations. This is the strongest additional experiment for C2.
4. Run offline scale-separation N3 on flagship. Extend to more runs only if the first result is clean.
5. Compute E3 Particle Life++ metrics. Add minimal regularizer baseline only if the current text claims Particle Life++ transfer strongly.
6. Do not run 9 NN-OEE optimizations unless the paper explicitly needs a statistical head-to-head with NN-OEE. With current claims, NN-OEE exemplar is enough as reference.

---

# 11. Experiments not recommended as main evidence

Do not use these as primary experiments:

- object/cell tracking via connected components, SAM, or cell-detection libraries;
- color-based species counts except in manually validated case studies;
- handcrafted video patch-token motifs from rendered RGB as main ecological descriptors;
- exhaustive hyperparameter robustness over every window size, patch size, $K$, seed, and frame stride;
- large NN-OEE optimization batch unless the claim is changed to a statistical NN-OEE-vs-MSPD benchmark;
- Boids as full ecological transfer claim. If included, use it as a limited-expressivity substrate.

---

# 12. Minimal final claim set after these experiments

If the planned experiments succeed, the paper can defend the following claims:

C1: MSPD optimization raises trajectory-distributional complexity relative to matched random controls under a selection-adjusted $\tau$ protocol.

C2: $\Delta H$ peaks identify transition-sensitive states; in validated ecological simulations these peaks align with visible turnover events.

C3: In high-complexity simulations, MSPD-selected temporal scales can coincide with independently measured scales of stable, spatially coherent dynamical role separation.

C4: deferred. Do not claim quantitative ecological richness in the current main experiment set. If revived, use calibrated semantic phenotype evaluation rather than handcrafted biodiversity proxies.

C5: MSPD-optimized systems exhibit increased blockwise frustration relative to matched random controls, especially on snapshot/representation axes.

C6: The MSPD protocol transfers to Particle Life++ under the same analysis pipeline; emergence depends on substrate expressivity and escape from zero-motion basins.