import re
import json


def json2str(obj: list | dict, indent: int = 4, current_indent: int = 0) -> str:
    """Converts JSON data to a string with controlled expansion depth.

    Args:
        obj: The JSON data.
        indent: The number of spaces to use for indentation.
        current_indent: The current indentation level.

    Returns:
        The formatted JSON string.
    """
    space = " " * current_indent
    next_space = " " * (current_indent + indent)

    if isinstance(obj, dict):
        items = []
        for key, value in obj.items():
            items.append(next_space + json.dumps(key, ensure_ascii=False) + ": " + json2str(value, indent, current_indent + indent))
        return "{\n" + ",\n".join(items) + "\n" + space + "}"
    elif isinstance(obj, list):
        if all(not isinstance(x, (list, dict)) for x in obj):
            return json.dumps(obj, ensure_ascii=False)
        else:
            items = []
            for item in obj:
                items.append(next_space + json2str(item, indent, current_indent + indent))
            return "[\n" + ",\n".join(items) + "\n" + space + "]"
    else:
        return json.dumps(obj, ensure_ascii=False)


def extract_variables(string: str, template: str) -> dict[str, str]:
    """Extracts variable values from a given string based on a template.

    Args:
        string (str): The actual string containing the values of the template variables.
        template (str): The template string containing variable names enclosed in curly braces.

    Returns:
        A dictionary containing variable names and their corresponding values.

    Example:
    >>> template = "{a} + {b} = {c}"
    >>> string = "1 + 2 = 3"
    >>> extract_variables(string, template)
    {'a': '1', 'b': '2', 'c': '3'}
    """
    variable_names = re.findall(r"\{(\w+)\}", template)
    regex_pattern = re.escape(template)
    for var in variable_names:
        regex_pattern = regex_pattern.replace(r"\{" + var + r"\}", r"(?P<" + var + r">[\s\S]+?)")
    match = re.match(regex_pattern, string)
    if not match:
        return {}
    return match.groupdict()
