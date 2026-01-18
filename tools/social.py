# tools/social.py
import time
from typing import List, Optional, Dict

from core.social.manager import UserManager
from core.social.schema import UserProfile
from core.tool_manager.registry import register


# =========================
# 内部辅助函数
# =========================

def _clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


# =========================
# 工具函数
# =========================

@register()
async def social_register_user(
        platform: str,
        user_id: str,
        nickname: str,
        impression: str,
        relationship_tags: List[str],
        user_manager: UserManager
) -> str:
    """
    [主动社交] 将一个新用户录入到你的社交网络中。
    只有当你觉得这个人“值得长期记住”时才调用。

    Args:
        platform: 来源平台
        user_id: 平台内原始用户 ID
        nickname: 你对他的称呼
        impression: 初步印象
        relationship_tags: 初始关系标签
    """
    uid = user_manager.resolve_uid(platform, user_id)

    if await user_manager.get_user(uid):
        return (
            f"用户已存在：{nickname} (UID: {uid})\n"
            f"如需修改信息，请使用 social_update_info。"
        )

    profile = UserProfile(
        uid=uid,
        platform=platform,
        user_id=user_id,
        nickname=nickname,
        impression=impression,
        relationship_tags=list(set(relationship_tags)) if relationship_tags else []
    )

    # 初始社会属性
    profile.favorability = 10.0
    profile.trust = 0.0
    profile.intimacy = 0.0
    profile.meta.setdefault("history", [])

    await user_manager.save_user(profile)

    return (
        f"已新建社交档案\n"
        f"- 昵称：{nickname}\n"
        f"- UID：{uid}\n"
        f"- 标签：{profile.relationship_tags}"
    )


@register()
async def social_update_perception(
        uid: str,
        dimension: str,
        delta: float,
        reason: str,
        user_manager: UserManager
) -> str:
    """
    更新你对某个用户的情感 / 认知维度。

    Args:
        uid: 用户唯一标识
        dimension: favorability | trust | intimacy
        delta: 变化值（建议 -10 ~ +10）
        reason: 变更原因（会写入历史）
    """
    profile = await user_manager.get_user(uid)
    if not profile:
        return f"未找到用户 (UID: {uid})，请先注册。"

    # 维度配置表
    DIMENSIONS: Dict[str, Dict] = {
        "favorability": {"min": -100.0, "max": 100.0, "label": "好感度"},
        "trust": {"min": 0.0, "max": 100.0, "label": "信任度"},
        "intimacy": {"min": 0.0, "max": 100.0, "label": "亲密度"},
    }

    if dimension not in DIMENSIONS:
        return (
            "无效的维度。\n"
            "可选：favorability（好感）, trust（信任）, intimacy（亲密）"
        )

    conf = DIMENSIONS[dimension]
    old_value = getattr(profile, dimension, 0.0)
    new_value = _clamp(old_value + delta, conf["min"], conf["max"])
    setattr(profile, dimension, new_value)

    # 记录历史（结构化，方便未来分析）
    profile.meta.setdefault("history", []).append({
        "time": time.time(),
        "dimension": dimension,
        "delta": delta,
        "from": old_value,
        "to": new_value,
        "reason": reason,
    })

    await user_manager.save_user(profile)

    return (
        f"已更新 {profile.nickname} 的{conf['label']}\n"
        f"- 变化：{old_value:.1f} → {new_value:.1f} ({delta:+.1f})\n"
        f"- 原因：{reason}"
    )


@register()
async def social_update_info(
        user_manager: UserManager,
        uid: str,
        nickname: Optional[str] = None,
        impression: Optional[str] = None,
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
) -> str:
    """
    更新用户的基础信息（昵称 / 印象 / 标签）。
    """
    profile = await user_manager.get_user(uid)
    if not profile:
        return f"未找到用户 (UID: {uid})"

    logs: List[str] = []

    if nickname and nickname != profile.nickname:
        logs.append(f"昵称：{profile.nickname} → {nickname}")
        profile.nickname = nickname

    if impression and impression != profile.impression:
        profile.impression = impression
        logs.append("印象已更新")

    if add_tags:
        added = [t for t in add_tags if t not in profile.relationship_tags]
        if added:
            profile.relationship_tags.extend(added)
            logs.append(f"添加标签：{added}")

    if remove_tags:
        removed = [t for t in remove_tags if t in profile.relationship_tags]
        if removed:
            profile.relationship_tags = [
                t for t in profile.relationship_tags if t not in removed
            ]
            logs.append(f"移除标签：{removed}")

    if not logs:
        return "未检测到任何需要更新的内容。"

    await user_manager.save_user(profile)

    return "用户信息已更新：\n" + "\n".join(f"- {l}" for l in logs)


@register()
async def social_lookup(uid: str, user_manager: UserManager) -> str:
    """查询用户完整档案"""
    profile = await user_manager.get_user(uid)
    if not profile:
        return "用户不存在。"
    return str(profile.to_dict())


@register()
async def social_list_users(
        user_manager: UserManager,
        limit: int = 20,
        offset: int = 0,
) -> str:
    """
    [查询工具] 列出社交名册中的用户（分页）。
    """
    users = await user_manager.list_users(limit, offset)

    if not users:
        return "社交名册为空。" if offset == 0 else "没有更多用户了。"

    lines: List[str] = [f"社交名册（显示 {len(users)} 位，offset={offset}）"]

    for u in users:
        last_seen = time.strftime("%Y-%m-%d %H:%M", time.localtime(u.last_seen))

        tags = f"[{', '.join(u.relationship_tags)}]" if u.relationship_tags else ""
        stats = f"好感 {u.favorability:.0f} | 信任 {u.trust:.0f}"

        lines.append(
            f"- {u.nickname or '(无昵称)'} `{u.uid}` {tags}\n"
            f"  {stats} ｜ 上次互动：{last_seen}"
        )

        if u.impression:
            lines.append(f"  *印象：{u.impression[:40]}*")

    return "\n".join(lines)
