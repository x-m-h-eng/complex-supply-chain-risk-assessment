import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit_authenticator as stauth
import networkx as nx

from core.graph_engine import (
    load_json, build_flow_graph, eval_state, value_function,
    shapley_mc, shapley_hierarchical, normalize_phi, sensitivity_scan,
)
from core.text_agents import run_pipeline, node_exposure
from core.explainer import build_context, explain, guard

st.set_page_config(page_title="复杂供应链风险全景测度", page_icon="🕸️", layout="wide")
# === 登录验证模块开始 ===
credentials = st.secrets["credentials"].to_dict()
cookie = st.secrets["cookie"]

authenticator = stauth.Authenticate(
    credentials,
    cookie["name"],
    cookie["key"],
    cookie["expiry_days"],
)

authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout(location="sidebar")
    # 登录成功，下面的代码正常执行
    pass 
elif st.session_state["authentication_status"] is False:
    st.error("❌ 用户名或密码错误")
    st.info("💡 提示：如果登录失败或页面卡顿，请多刷新几次浏览器重试。")
    st.stop()
elif st.session_state["authentication_status"] is None:
    st.warning("请先登录")
    st.caption("若页面未能正常显示登录框，请刷新浏览器重试。")
    st.stop()
# === 登录验证模块结束 ===

CFG = load_json("config/topology_13.json")
EVENTS = load_json("config/events.json")
PLAYERS = [n["id"] for n in CFG["nodes"]]
NODE_MAP = {n["id"]: n for n in CFG["nodes"]}
WINDOW_END = CFG["meta"]["window"][1]


# ------------------------------------------------------------------ 计算（缓存）
@st.cache_data(show_spinner=False)
def compute_all(n_perm: int, seed: int):
    base = eval_state(CFG)
    if len(PLAYERS) <= 10:
        phi_raw = shapley_mc(CFG, PLAYERS, base, n_perm, seed)
        method = "蒙特卡罗排列采样 Shapley"
    else:
        phi_raw = shapley_hierarchical(CFG, base, n_perm, seed)
        method = "分层 Shapley（候选集 + 背景聚合）"

    phi = normalize_phi(phi_raw)

    results = run_pipeline(EVENTS, WINDOW_END)
    exposure = node_exposure(results)

    G = build_flow_graph(CFG)
    und = G.to_undirected()
    cut_points = set(nx.articulation_points(und)) if und.number_of_nodes() > 2 else set()
    bridges = {u for e in nx.bridges(und) for u in e} if und.number_of_nodes() > 2 else set()

    rows = []
    ranked = sorted(phi.items(), key=lambda x: -x[1])
    rank_map = {k: i + 1 for i, (k, _) in enumerate(ranked)}

    for n in CFG["nodes"]:
        nid = n["id"]
        exp = exposure.get(nid, 0.0)
        resp = phi.get(nid, 0.0)
        if exp >= 0.5 and resp >= 0.5:
            quad = "显性核心"
        elif exp < 0.5 and resp >= 0.5:
            quad = "隐性关键"
        elif exp >= 0.5 and resp < 0.5:
            quad = "局部扰动"
        else:
            quad = "常规节点"

        anchors = [r for r in results if r["node"] == nid and not r.get("rejected")]
        rows.append({
            "id": nid, "name": n["name"], "type": n["type"], "role": n["role"],
            "capacity_source": n["capacity_source"],
            "exposure": round(exp, 4),
            "responsibility": round(resp, 4),
            "rank": rank_map[nid],
            "quadrant": quad,
            "in_degree": G.in_degree(f"{nid}__in"),
            "out_degree": G.out_degree(f"{nid}__out"),
            "is_cut_point": nid in cut_points,
            "is_bridge": nid in bridges,
            "anchors": [a["evidence_anchor"] for a in anchors],
            "rule_log": [lg for a in anchors for lg in a["rule_log"]],
            "critical": n.get("critical", False),
            "demand_weight": CFG["demand_weight"],
            "penalty": CFG["penalty"],
        })

    df = pd.DataFrame(rows).sort_values("responsibility", ascending=False).reset_index(drop=True)
    return df, results, method


def get_api_key() -> str:
    try:
        return st.secrets.get("DEEPSEEK_API_KEY", "")
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY", "")


# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.title("🕸️ 复杂供应链风险全景测度")
    st.caption("暴露度 — 责任度 二维交叉决策矩阵")
    n_perm = st.slider("蒙特卡罗采样次数", 100, 2000, 500, 100)
    seed = st.number_input("随机种子", 1, 9999, 42)
    st.divider()
    api_key_input = st.text_input("DeepSeek API Key（可选，仅用于解释层）",
                                  type="password", value=get_api_key())
    st.divider()
    st.caption("数据边界：容量为公开锚点 / 相对比例 / 业务先验三级精度映射，"
               "不代表企业内部绝对数据。")

df, results, method = compute_all(int(n_perm), int(seed))

# ------------------------------------------------------------------ 顶部指标
c1, c2, c3, c4 = st.columns(4)
c1.metric("节点数", len(df))
c2.metric("显性核心", int((df["quadrant"] == "显性核心").sum()))
c3.metric("隐性关键", int((df["quadrant"] == "隐性关键").sum()))
c4.metric("归因方法", method.split("（")[0])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 二维决策矩阵", "② 责任度归因", "③ 文本测度与证据链", "④ 敏感性分析", "⑤ 智能体问答"]
)

# ------------------------------------------------------------------ ① 矩阵
with tab1:
    st.subheader("暴露度 — 责任度 二维交叉矩阵")
    fig = px.scatter(
        df, x="exposure", y="responsibility",
        color="quadrant", text="name", size=[0.06] * len(df),
        hover_data=["id", "role", "capacity_source", "rank"],
        color_discrete_map={
            "显性核心": "#d62728", "隐性关键": "#ff7f0e",
            "局部扰动": "#1f77b4", "常规节点": "#7f7f7f",
        },
        labels={"exposure": "暴露度（文本测度）", "responsibility": "责任度（Shapley 归因）"},
        range_x=[0, 1], range_y=[0, 1],
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center", marker=dict(size=16))
    fig.update_layout(height=560, legend_title="象限")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
- **显性核心（双高）**：舆情热 + 破坏力大，优先保供。
- **隐性关键（低暴露—高责任）**：潜伏炸弹，平时无预警但一断全网瘫，需物理冗余 / 备产线 / 缓冲库存。
- **局部扰动（高暴露—低责任）**：虚假警报，末端可替代，不应占用应急预算。
- **常规节点**：常态巡检。
""")

# ------------------------------------------------------------------ ② 责任度
with tab2:
    st.subheader("Shapley 责任度排序（Top-K）")
    top_k = st.slider("显示 Top-K", 3, len(df), min(10, len(df)))
    d = df.head(top_k).iloc[::-1]
    fig2 = px.bar(d, x="responsibility", y="name", orientation="h",
                  color="quadrant", text="responsibility",
                  color_discrete_map={
                      "显性核心": "#d62728", "隐性关键": "#ff7f0e",
                      "局部扰动": "#1f77b4", "常规节点": "#7f7f7f",
                  })
    fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig2.update_layout(height=460, yaxis_title="", xaxis_title="归一化责任度")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        df[["rank", "id", "name", "type", "role", "exposure", "responsibility",
            "quadrant", "is_cut_point", "capacity_source"]],
        use_container_width=True, hide_index=True,
    )

# ------------------------------------------------------------------ ③ 文本测度
with tab3:
    st.subheader("多智能体文本测度：Extractor → Feedback → Revision")
    rows = []
    for r in results:
        rows.append({
            "事件": r["title"], "节点": NODE_MAP[r["node"]]["name"],
            "日期": r["date"], "范围词": r["scope_word"],
            "raw": r["severity_raw"],
            "final": r["severity_final"] if not r.get("rejected") else "驳回",
            "证据锚点": r["evidence_anchor"],
            "规则日志": " | ".join(r["rule_log"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info("R1 为硬门禁：证据锚点必须在原文逐字存在，否则直接驳回，不进入后续规则。"
            "所有暴露度评分均可追溯至原文锚点与触发规则。")

# ------------------------------------------------------------------ ④ 敏感性
with tab4:
    st.subheader("容量敏感性扫描（代理 / 先验节点 ±20%–30%）")
    if st.button("运行敏感性扫描", type="primary"):
        with st.spinner("扫描中…"):
            runs = sensitivity_scan(CFG, PLAYERS, n_perm=min(int(n_perm), 300), seed=int(seed))
        base = runs["baseline"]
        keys = sorted(base, key=lambda k: -base[k])[:8]
        mat = pd.DataFrame({k: [v[x] for x in keys] for k, v in runs.items()},
                           index=[NODE_MAP[x]["name"] for x in keys])
        st.dataframe(mat.round(4), use_container_width=True)
        st.line_chart(mat)
        st.caption("若核心节点排序在各扰动情形下不翻转，则责任度结论稳健。")

# ------------------------------------------------------------------ ⑤ 智能体
with tab5:
    st.subheader("DeepSeek 业务翻译官（强约束 JSON Context）")
    target = st.selectbox("选择要追问的节点",
                          df["id"] + " " + df["name"], index=0)
    tid = target.split(" ")[0]
    row = df[df["id"] == tid].iloc[0].to_dict()
    top_nodes = df.head(5)[["id", "name", "responsibility", "quadrant"]].to_dict("records")
    ctx = build_context(row, top_nodes)

    with st.expander("查看强约束 JSON Context（大模型唯一可见事实）"):
        st.json(ctx)

    question = st.text_area(
        "管理者问题",
        value=f"{row['name']}如果停产，对全局供应保障影响有多大？该不该优先投预算？",
        height=90,
    )
    if st.button("生成解释与决策建议", type="primary"):
        ok, reason = guard(question)
        if not ok:
            st.warning(f"越界拦截：{reason}")
        elif not api_key_input:
            st.error("请在侧边栏填写 DeepSeek API Key（或配置 secrets.toml）。")
        else:
            with st.spinner("解释层生成中…"):
                ans = explain(api_key_input, ctx, question)
            st.markdown(ans)
            st.caption("【可验证性】本回答的数值均来自后端图引擎，可追溯至 Shapley 值与原文证据锚点。")

st.divider()
st.caption("离线重加工 · 在线轻交互 | 后端不做 LLM 数值计算，LLM 不做后端数值计算")