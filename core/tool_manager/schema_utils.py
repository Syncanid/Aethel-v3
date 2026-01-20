# core/tool_manager/schema_utils.py
import inspect
import re
from typing import Dict, Any, Union, get_origin, get_args, Literal


class SchemaGenerator:
    @staticmethod
    def get_function_schema(func, func_name: str) -> Dict[str, Any]:
        docstring = inspect.getdoc(func) or ""
        parsed_doc = SchemaGenerator._parse_google_docstring(docstring)
        description = parsed_doc["description"] or "No description available"

        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        signature = inspect.signature(func)
        for param_name, param in signature.parameters.items():
            # 跳过依赖注入的参数 (self, cls, 以及后续在 Aggregator 中定义的注入参数)
            # 这里暂时只跳过 self/cls，具体的依赖剔除在 Aggregator 中处理
            if param_name in ("self", "cls"):
                continue

            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)

            param_desc = parsed_doc["args"].get(param_name, "")
            param_schema = SchemaGenerator._get_param_schema(param, param_desc)
            parameters["properties"][param_name] = param_schema

        return {
            "type": "function",
            "function": {
                "name": func_name,
                "description": description,
                "parameters": parameters,
            }
        }

    @staticmethod
    def _get_param_schema(param: inspect.Parameter, description: str) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"description": description or ""}
        if param.annotation != inspect.Parameter.empty:
            schema.update(SchemaGenerator._map_type_to_schema(param.annotation))
        else:
            schema["type"] = "string"

        # 处理 Literal 枚举
        if hasattr(param.annotation, "__origin__") and param.annotation.__origin__ is Literal:
            schema["enum"] = list(param.annotation.__args__)
        return schema

    @staticmethod
    def _map_type_to_schema(annotation: Any) -> Dict[str, Any]:
        # 处理 Optional[T]
        if get_origin(annotation) is Union:
            args = get_args(annotation)
            if type(None) in args:
                non_null_types = [t for t in args if t is not type(None)]
                if len(non_null_types) == 1:
                    return SchemaGenerator._map_type_to_schema(non_null_types[0])

        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array", "items": {"type": "string"}},
            dict: {"type": "object"},
            Any: {"type": "string"},
        }

        if annotation in type_map:
            return type_map[annotation].copy()

        # 泛型处理 List[T], Dict[K,V]
        origin = get_origin(annotation)
        if origin:
            args = get_args(annotation)
            if origin is list or origin.__name__ == "list":
                item_schema = {"type": "string"}
                if args:
                    item_schema = SchemaGenerator._map_type_to_schema(args[0])
                return {"type": "array", "items": item_schema}
            if origin is dict or origin.__name__ == "dict":
                return {"type": "object"}

        # 回退逻辑
        type_str = str(annotation).lower()
        if "int" in type_str: return {"type": "integer"}
        if "float" in type_str: return {"type": "number"}
        if "bool" in type_str: return {"type": "boolean"}
        if "dict" in type_str: return {"type": "object"}
        if "list" in type_str: return {"type": "array", "items": {"type": "string"}}

        return {"type": "string"}

    @staticmethod
    def _parse_google_docstring(docstring: str) -> Dict[str, Any]:
        if not docstring:
            return {"description": "", "args": {}}

        parts = re.split(r'\n\s*\n', docstring.strip(), maxsplit=1)
        description = parts[0] if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        args_desc = {}
        args_match = re.search(r'Args:\s*\n((?:\s{4,}.*\n)+)', rest, re.MULTILINE)
        if args_match:
            args_block = args_match.group(1).strip()
            param_blocks = re.split(r'\n\s{2,}(?=[a-zA-Z_])', args_block)
            for block in param_blocks:
                if not block.strip(): continue
                match = re.match(r'(\w+)\s*(?:\(([^)]+)\))?:\s*(.*)', block, re.DOTALL)
                if match:
                    param_name = match.group(1).strip()
                    desc = match.group(3).strip()
                    desc = re.sub(r'\n\s+', ' ', desc).strip()
                    args_desc[param_name] = desc

        return {"description": description.strip(), "args": args_desc}
