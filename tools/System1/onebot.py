# tools/onebot.py
import datetime
import logging
from typing import Any, Dict, List

from core.infrastructure.config_loader import Config
from core.io.adapters.onebot_v11 import OneBotV11Adapter
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


# =========================
# 辅助函数
# =========================

def _format_date(timestamp: int) -> str:
    if not timestamp or timestamp <= 0:
        return '未知'
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone(datetime.timedelta(hours=8)))
    return dt.strftime('%Y-%m-%d %H:%M:%S')


# =========================
# 工具定义
# =========================

@register()
async def get_friends(ob_adapter: OneBotV11Adapter, config: Config = None) -> List[Dict[str, Any]]:
    """获取好友列表"""
    resp = await ob_adapter.call_api("get_friend_list")
    if not resp or resp.get("status") != "ok":
        return []

    data = resp['data']
    friends = []

    # 获取自身 ID 避免包含自己
    self_id = int(config.get("system.bot_self_id", 0)) if config else 0

    for friend in data:
        if friend["user_id"] == self_id:
            continue

        birthday = ""
        if friend["birthday_year"] > 0:
            birthday = f"{friend['birthday_year']}年"
        if friend["birthday_day"] > 0 and friend["birthday_month"] > 0:
            birthday += f"{friend['birthday_month']}月{friend['birthday_day']}日"

        friends.append({
            "name": friend.get("remark") or friend.get("nickname"),
            "user_id": friend["user_id"],
            "sex": friend.get("sex", "unknown")
        })
    return friends


@register()
async def get_user_info(ob_adapter: OneBotV11Adapter, user_id: str) -> str:
    """获取指定 QQ 用户的详细资料。"""

    def format_user_data(data: dict) -> str:
        lines = []

        def format_date(timestamp: int) -> str:
            if not timestamp or timestamp <= 0:
                return '未知'
            # 转换为北京时间（UTC+8）
            dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone(datetime.timedelta(hours=8)))
            return dt.strftime('%Y-%m-%d %H:%M:%S')

        # ===== 基础身份信息 =====
        lines.append(f"昵称：{data.get('nick') or '未设置'}")
        long_nick = data.get('longNick', '').strip()
        if long_nick:
            lines.append(f"个性签名：{long_nick}")
        lines.append(f"QQ号：{data.get('uin', '未知')}")
        if data.get('qid'):
            lines.append(f"自定义ID（QID）：{data['qid']}")
        lines.append(f"UID：{data.get('uid', '未知')}")

        # ===== 性别 =====
        sex_map = {'male': '男', 'female': '女', 'unknown': '未知'}
        sex = data.get('sex')
        lines.append(f"性别：{sex_map.get(sex, '未设置')}")

        # ===== 生日与年龄 =====
        birthday_year = data.get('birthday_year', 0)
        birthday_month = data.get('birthday_month', 0)
        birthday_day = data.get('birthday_day', 0)

        if birthday_year > 0 and birthday_month > 0 and birthday_day > 0:
            lines.append(f"生日：{birthday_year}年{birthday_month}月{birthday_day}日")
        elif birthday_month > 0 and birthday_day > 0:
            lines.append(f"生日：{birthday_month}月{birthday_day}日")
        elif birthday_year > 0:
            lines.append(f"生日：{birthday_year}年")
        else:
            lines.append("生日：未填写")

        age = data.get('age')
        lines.append(f"年龄：{age if age is not None else '未知'}")

        # ===== 星座 =====
        constellation_map = {
            1: '白羊座', 2: '金牛座', 3: '双子座', 4: '巨蟹座',
            5: '狮子座', 6: '处女座', 7: '天秤座', 8: '天蝎座',
            9: '射手座', 10: '摩羯座', 11: '水瓶座', 12: '双鱼座'
        }
        constellation = data.get('constellation', 0)
        constellation_str = constellation_map.get(constellation, '未知')
        lines.append(f"星座：{constellation_str}")

        # ===== 生肖 =====
        sheng_xiao_map = {
            1: '鼠', 2: '牛', 3: '虎', 4: '兔',
            5: '龙', 6: '蛇', 7: '马', 8: '羊',
            9: '猴', 10: '鸡', 11: '狗', 12: '猪'
        }
        sheng_xiao = sheng_xiao_map.get(data.get('shengXiao'), '未知')
        lines.append(f"生肖：{sheng_xiao}")

        # ===== 血型 =====
        blood_type_map = {1: 'A型', 2: 'B型', 3: 'O型', 4: 'AB型'}
        k_blood_type = data.get('kBloodType')
        if isinstance(k_blood_type, int) and 1 <= k_blood_type <= 4:
            blood = blood_type_map[k_blood_type]
        else:
            blood = '未填写'
        lines.append(f"血型：{blood}")

        # ===== 地区信息 =====
        country = data.get('country')
        province = data.get('province')
        city = data.get('city')
        if country:
            if province in {'香港', '澳门', '台湾'}:
                location = f"中国 {province}"
            else:
                location = ' '.join(filter(None, [country, province, city]))
        else:
            location = ''
        lines.append(f"所在地：{location or '未填写'}")

        # ===== 账号注册与活跃 =====
        reg_time = data.get('regTime')
        lines.append(f"注册时间：{format_date(reg_time)}")

        # ===== QQ等级 =====
        is_hide_qq_level = data.get('isHideQQLevel')
        qq_level = data.get('qqLevel', 0)
        if is_hide_qq_level:
            level_display = '（已隐藏）'
        else:
            level_display = str(qq_level) if qq_level > 0 else '0'
        lines.append(f"QQ等级：{level_display}")

        # ===== VIP状态 =====
        is_vip = data.get('is_vip')
        if is_vip:
            is_years_vip = data.get('is_years_vip')
            vip_type = '年费VIP' if is_years_vip else '普通VIP'
            vip_level = data.get('vip_level', '未知')
            lines.append(f"VIP状态：{vip_type}（等级 {vip_level}）")
        else:
            lines.append("VIP状态：否")

        # ===== 联系方式 =====
        email = data.get('eMail')
        if email and email not in {'-', ''}:
            lines.append(f"邮箱：{email}")

        phone_num = data.get('phoneNum')
        if phone_num and phone_num not in {'-', ''}:
            lines.append(f"手机号：{phone_num}")

        # ===== 兴趣与标签 =====
        interest = data.get('interest', '').strip()
        if interest:
            lines.append(f"兴趣爱好：{interest}")

        labels = data.get('labels')
        if isinstance(labels, list) and labels:
            lines.append(f"个人标签：{'、'.join(map(str, labels))}")

        # ===== 返回结果 =====
        return '\n'.join(lines)

    payload = {"user_id": user_id}
    resp = await ob_adapter.call_api("get_stranger_info", payload)

    if not resp or resp.get("status") != "ok":
        return f"获取用户 {user_id} 信息失败。"

    data = resp['data']
    return format_user_data(data)


@register()
async def get_groups(ob_adapter: OneBotV11Adapter) -> List[Dict[str, Any]]:
    """获取加入的群聊列表"""
    resp = await ob_adapter.call_api("get_group_list")
    if not resp or resp.get("status") != "ok":
        return []

    data = resp['data']
    groups = []
    for group in data:
        groups.append({
            "name": group.get("group_remark") or group.get("group_name"),
            "group_id": group["group_id"],
            "member_count": group["member_count"],
        })
    return groups


@register()
async def get_group_info(ob_adapter: OneBotV11Adapter, group_id: str) -> Dict[str, Any]:
    """获取特定群聊的详细信息"""
    payload = {"group_id": group_id}
    resp = await ob_adapter.call_api("get_group_info", payload)

    if not resp or resp.get("status") != "ok":
        return {"error": "获取失败"}

    data = resp['data']
    return {
        "name": data.get("group_name"),
        "remark": data.get("group_remark"),
        "group_id": data.get("group_id"),
        "member_count": data.get("member_count"),
        "max_member_count": data.get("max_member_count"),
    }


@register()
async def get_group_members(ob_adapter: OneBotV11Adapter, group_id: str) -> List[Dict[str, Any]]:
    """获取群员列表"""
    payload = {"group_id": group_id}
    resp = await ob_adapter.call_api("get_group_member_list", payload)
    data = resp['data']
    members = []
    for member in data:
        members.append({
            "name": member["card"] or member["nickname"],
            "user_id": member["user_id"],
            "sex": member["sex"],
            "title": member["title"],
            # "age": member["age"] if member["age"] > 0 else "未知",
            "level": member["level"],
            "role": member["role"],
            "is_robot": member["is_robot"],
        })
    return members


@register()
async def handle_friend_add_request(ob_adapter: OneBotV11Adapter, flag: str, approve: bool = True, remark: str = "") -> \
        Dict[str, Any]:
    """
    处理加好友请求
    :param flag: 加好友请求的 flag (需从上报事件中获取)
    :param approve: 是否同意请求，默认为 True
    :param remark: 同意后的好友备注
    """
    payload = {
        "flag": flag,
        "approve": approve,
        "remark": remark
    }
    resp = await ob_adapter.call_api("set_friend_add_request", payload)

    if not resp or resp.get("status") != "ok":
        return {"status": "error", "message": resp.get("message", "处理好友请求失败")}

    return {"status": "success", "message": "好友请求处理成功"}


@register()
async def delete_friend(ob_adapter: OneBotV11Adapter, user_id: str, temp_block: bool = False,
                        temp_both_del: bool = False) -> Dict[str, Any]:
    """
    删除好友（支持拉黑和双向删除）
    :param user_id: 目标 QQ 号
    :param temp_block: 是否加入黑名单 (拉黑)
    :param temp_both_del: 是否双向删除
    """
    payload = {
        "user_id": user_id,
        "temp_block": temp_block,
        "temp_both_del": temp_both_del
    }
    resp = await ob_adapter.call_api("delete_friend", payload)

    if not resp or resp.get("status") != "ok":
        return {"status": "error", "message": resp.get("message", f"删除好友 {user_id} 失败")}

    return {"status": "success", "message": f"成功删除好友 {user_id}"}


@register()
async def handle_group_add_request(ob_adapter: OneBotV11Adapter, flag: str, sub_type: str, approve: bool = True,
                                   reason: str = "") -> Dict[str, Any]:
    """
    处理加群请求或邀请
    :param flag: 请求 flag (需从上报事件中获取)
    :param sub_type: 请求类型 ('add' 为加群申请, 'invite' 为被邀请加入)
    :param approve: 是否同意
    :param reason: 拒绝时的理由
    """
    if sub_type not in ["add", "invite"]:
        return {"status": "error", "message": "sub_type 必须为 'add' 或 'invite'"}

    payload = {
        "flag": flag,
        "sub_type": sub_type,
        "approve": approve,
        "reason": reason
    }
    resp = await ob_adapter.call_api("set_group_add_request", payload)

    if not resp or resp.get("status") != "ok":
        return {"status": "error", "message": resp.get("message", "处理群请求失败")}

    return {"status": "success", "message": "群请求处理成功"}


@register()
async def leave_group(ob_adapter: OneBotV11Adapter, group_id: str, is_dismiss: bool = False) -> Dict[str, Any]:
    """
    退出或解散群聊
    :param group_id: 群号
    :param is_dismiss: 是否解散群聊 (设为 True 需具备群主权限，否则为普通退群)
    """
    payload = {
        "group_id": group_id,
        "is_dismiss": is_dismiss
    }
    resp = await ob_adapter.call_api("set_group_leave", payload)

    if not resp or resp.get("status") != "ok":
        action = "解散" if is_dismiss else "退出"
        return {"status": "error", "message": resp.get("message", f"{action}群 {group_id} 失败")}

    return {"status": "success", "message": f"成功{'解散' if is_dismiss else '退出'}群 {group_id}"}


@register()
async def kick_group_members(ob_adapter: OneBotV11Adapter, group_id: str, user_ids: List[str],
                             reject_add_request: bool = False) -> Dict[str, Any]:
    """
    批量踢出群成员
    :param group_id: 群号
    :param user_ids: 待踢出的 QQ 号列表
    :param reject_add_request: 是否拒绝被踢出者的后续加群请求 (相当于群内拉黑)
    """
    if not user_ids:
        return {"status": "error", "message": "未提供需要踢出的用户列表"}

    payload = {
        "group_id": group_id,
        "user_id": user_ids,
        "reject_add_request": reject_add_request
    }
    resp = await ob_adapter.call_api("set_group_kick_members", payload)

    if not resp or resp.get("status") != "ok":
        return {"status": "error", "message": resp.get("message", "批量踢出成员失败")}

    return {"status": "success", "message": f"成功从群 {group_id} 踢出 {len(user_ids)} 名成员"}
