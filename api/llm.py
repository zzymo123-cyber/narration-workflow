import asyncio
import json
import os
import httpx
import anthropic

DEFAULT_BASE_URL = "https://api.idealab.com/v1"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_LLM_PROVIDER = "idealab"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def _read_settings() -> dict:
    try:
        from api.routes.settings import read_settings
        return read_settings()
    except Exception:
        return {}


def _get_llm_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", "")
    if not provider:
        provider = _read_settings().get("llm_provider", "")
    provider = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
    if provider == "ds":
        return "deepseek"
    return provider


def get_llm_config() -> dict:
    """Return resolved LLM config for diagnostics. Never exposes full API key."""
    provider = _get_llm_provider()
    settings = _read_settings()

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or settings.get("deepseek_api_key", "")
        base_url = os.environ.get("DEEPSEEK_BASE_URL") or settings.get("deepseek_base_url", "") or DEFAULT_DEEPSEEK_BASE_URL
        model = os.environ.get("DEEPSEEK_MODEL") or settings.get("deepseek_model", "") or DEFAULT_DEEPSEEK_MODEL
    else:
        api_key = os.environ.get("IDEALAB_API_KEY") or settings.get("idealab_api_key", "")
        base_url = os.environ.get("IDEALAB_BASE_URL") or settings.get("idealab_base_url", "") or DEFAULT_BASE_URL
        model = DEFAULT_MODEL

    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "has_api_key": bool(api_key),
        "api_key_tail": api_key[-4:] if len(api_key) >= 4 else (api_key if api_key else ""),
    }


def _get_client(api_key: str) -> anthropic.Anthropic:
    base_url = os.environ.get("IDEALAB_BASE_URL", "")
    if not base_url:
        base_url = _read_settings().get("idealab_base_url", "") or DEFAULT_BASE_URL
    # *.alibaba-inc.com 在 Windows ProxyOverride 直连，httpx 不读 ProxyOverride，显式绕过代理
    http_client = httpx.Client(trust_env=False, verify=False)
    return anthropic.Anthropic(api_key=api_key, base_url=base_url, http_client=http_client)


def _get_deepseek_config(fallback_api_key: str = "") -> tuple[str, str, str]:
    settings = _read_settings()
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or settings.get("deepseek_api_key", "")
        or fallback_api_key
    )
    base_url = (
        os.environ.get("DEEPSEEK_BASE_URL")
        or settings.get("deepseek_base_url", "")
        or DEFAULT_DEEPSEEK_BASE_URL
    )
    model = (
        os.environ.get("DEEPSEEK_MODEL")
        or settings.get("deepseek_model", "")
        or DEFAULT_DEEPSEEK_MODEL
    )
    return api_key, base_url, model


def _get_deepseek_client(api_key: str, base_url: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def _generate_deepseek_prompt(api_key: str, system: str, user_message: str, model: str | None = None) -> str:
    ds_key, base_url, ds_model = _get_deepseek_config(api_key)
    client = _get_deepseek_client(ds_key, base_url)
    resp = client.chat.completions.create(
        model=model or ds_model,
        max_tokens=10000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content or ""


def generate_prompt(api_key: str, system: str, user_message: str, model: str = DEFAULT_MODEL) -> str:
    """调用 LLM 生成提示词，返回纯文本"""
    if _get_llm_provider() == "deepseek":
        return _generate_deepseek_prompt(api_key, system, user_message, None)
    client = _get_client(api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=10000,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text


def chat_with_agent(
    api_key: str,
    messages: list[dict],
    system_prompt: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Agent 聊天。LLM 必须返回 JSON: {"reply": "...", "actions": [...]}
    如果返回非 JSON，将整个内容作为 reply，actions 为空。
    """
    if _get_llm_provider() == "deepseek":
        ds_key, base_url, ds_model = _get_deepseek_config(api_key)
        client = _get_deepseek_client(ds_key, base_url)
        resp = client.chat.completions.create(
            model=ds_model,
            max_tokens=10000,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
            return {
                "reply": data.get("reply", ""),
                "actions": data.get("actions", []),
                "_raw": content,
            }
        except (json.JSONDecodeError, AttributeError):
            return {"reply": content, "actions": [], "_raw": content}

    client = _get_client(api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=10000,
        system=system_prompt,
        messages=messages,
    )
    content = resp.content[0].text
    try:
        data = json.loads(content)
        return {
            "reply": data.get("reply", ""),
            "actions": data.get("actions", []),
            "_raw": content,
        }
    except (json.JSONDecodeError, AttributeError):
        return {"reply": content, "actions": [], "_raw": content}


# ── 异步版本 ──


async def generate_prompt_async(api_key: str, system: str, user_message: str, model: str = DEFAULT_MODEL) -> str:
    """异步调用 LLM 生成提示词"""
    return await asyncio.to_thread(generate_prompt, api_key, system, user_message, model)


async def chat_with_agent_async(
    api_key: str,
    messages: list[dict],
    system_prompt: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """异步 Agent 聊天"""
    return await asyncio.to_thread(chat_with_agent, api_key, messages, system_prompt, model)


def _resolve_api_key() -> str:
    """Resolve API key based on current provider setting."""
    provider = _get_llm_provider()
    settings = _read_settings()
    if provider == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY") or settings.get("deepseek_api_key", "")
    return os.environ.get("IDEALAB_API_KEY") or settings.get("idealab_api_key", "")


def test_llm_connection() -> dict:
    """Test LLM connection with a minimal prompt. Returns diagnostic result."""
    config = get_llm_config()
    api_key = _resolve_api_key()

    if not api_key:
        return {
            "success": False,
            **config,
            "error_type": "no_api_key",
            "error_message": "未配置 API Key",
        }

    try:
        generate_prompt(api_key, "Reply with OK", "test")
        return {"success": True, **config, "error_type": None, "error_message": None}
    except anthropic.AuthenticationError as e:
        return {
            "success": False, **config,
            "error_type": "auth_failed",
            "error_message": "API Key 无效或已过期",
        }
    except anthropic.NotFoundError as e:
        return {
            "success": False, **config,
            "error_type": "not_found",
            "error_message": f"模型或端点不存在: {e}",
        }
    except anthropic.PermissionDeniedError as e:
        return {
            "success": False, **config,
            "error_type": "permission_denied",
            "error_message": "无权访问该模型",
        }
    except httpx.ConnectError as e:
        return {
            "success": False, **config,
            "error_type": "connection_error",
            "error_message": f"无法连接到 {config['base_url']}",
        }
    except Exception as e:
        err_str = str(e)
        # Detect common error patterns
        if "401" in err_str or "Authentication" in err_str or "Invalid" in err_str:
            error_type = "auth_failed"
            error_message = "API Key 无效或认证失败"
        elif "403" in err_str or "Permission" in err_str:
            error_type = "permission_denied"
            error_message = "无权访问该模型"
        elif "404" in err_str or "Not Found" in err_str:
            error_type = "not_found"
            error_message = "模型或端点不存在"
        elif "Connection" in err_str or "connect" in err_str.lower():
            error_type = "connection_error"
            error_message = f"无法连接到 {config['base_url']}"
        else:
            error_type = "unknown"
            error_message = err_str[:200]
        return {
            "success": False, **config,
            "error_type": error_type,
            "error_message": error_message,
        }
