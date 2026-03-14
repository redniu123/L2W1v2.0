#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ast
from pathlib import Path

path = Path('modules/vlm_expert/constrained_prompter.py')
lines = path.read_text(encoding='utf-8').splitlines(keepends=True)

# Find broken method boundaries
start = next((i for i, l in enumerate(lines) if '# v5.1: Targeted Correction' in l), None)
end = next((i for i, l in enumerate(lines) if i > (start or 0) and 'def generate_boundary_prompt' in l), None)
print(f'start={start} end={end}')

# Build replacement using only ASCII source + chr() for Chinese
def c(s): return s  # identity, strings are already unicode in py3

nl = '\n'

new_method = (
    '    # v5.1: Targeted Correction (Soft-Hint)' + nl +
    '    def generate_targeted_correction_prompt(' + nl +
    '        self, T_A, min_conf_idx=None, domain=None, image_path=None' + nl +
    '    ):' + nl +
    '        if domain is None:' + nl +
    '            domain = "' + chr(0x5730) + chr(0x8d28) + chr(0x52d8) + chr(0x63a2) + '"' + nl +
    '        system_prompt = (' + nl +
    '            "' + chr(0x4f60) + chr(0x662f) + chr(0x4e00) + chr(0x4e2a) + chr(0x4e25) +
    chr(0x683c) + chr(0x7684) + ' OCR ' + chr(0x7ea0) + chr(0x9519) + chr(0x4e13) +
    chr(0x5bb6) + chr(0xff0c) + chr(0x4fee) + chr(0x6b63) + chr(0x5355) + chr(0x884c) +
    chr(0x6587) + chr(0x672c) + chr(0x4e2d) + chr(0x7684) + chr(0x8bc6) + chr(0x522b) +
    chr(0x9519) + chr(0x8bef) + chr(0xff0c) + chr(0x4fdd) + chr(0x7559) + chr(0x539f) +
    chr(0x6587) + chr(0x539f) + chr(0x8c8c) + chr(0x3002) + '"' + nl +
    '        )' + nl +
    '        hint_lines = []' + nl +
    '        if min_conf_idx is not None and 0 <= min_conf_idx < len(T_A):' + nl +
    '            hint_lines.append(' + nl +
    '                "' + chr(0x5176) + chr(0x4e2d) + chr(0x7b2c) + ' " + str(min_conf_idx + 1) + ' +
    '" ' + chr(0x4e2a) + chr(0x5b57) + chr(0x7b26) + chr(0x7684) + chr(0x673a) +
    chr(0x5668) + chr(0x7f6e) + chr(0x4fe1) + chr(0x5ea6) + chr(0x6781) + chr(0x4f4e) +
    chr(0xff0c) + chr(0x8bf7) + chr(0x91cd) + chr(0x70b9) + chr(0x5173) + chr(0x6ce8) +
    chr(0x3002) + '"' + nl +
    '            )' + nl +
    '        if domain:' + nl +
    '            hint_lines.append(' + nl +
    '                "' + chr(0x672c) + chr(0x6587) + chr(0x672c) + chr(0x5c5e) +
    chr(0x4e8e) + chr(0x3010) + '" + domain + "' + chr(0x3011) + chr(0x9886) +
    chr(0x57df) + chr(0xff0c) + chr(0x8bf7) + chr(0x7559) + chr(0x610f) + chr(0x4e13) +
    chr(0x4e1a) + chr(0x672f) + chr(0x8bed) + chr(0x7684) + chr(0x51c6) + chr(0x786e) +
    chr(0x6027) + chr(0x3002) + '"' + nl +
    '            )' + nl +
    '        sep = "\\n"' + nl +
    '        prefix = "' + chr(0x7cfb) + chr(0x7edf) + chr(0x68c0) + chr(0x6d4b) +
    chr(0x5230) + chr(0x8be5) + chr(0x6587) + chr(0x672c) + chr(0x53ef) + chr(0x80fd) +
    chr(0x5b58) + chr(0x5728) + chr(0x8bc6) + chr(0x522b) + chr(0x9519) + chr(0x8bef) +
    chr(0x3002) + '"' + nl +
    '        if hint_lines:' + nl +
    '            hint_block = prefix + "\\n" + "\\n".join(hint_lines) + "\\n\\n"' + nl +
    '        else:' + nl +
    '            hint_block = prefix + "\\n\\n"' + nl +
    '        user_prompt = (' + nl +
    '            "' + chr(0x4f60) + chr(0x662f) + chr(0x4e00) + chr(0x4e2a) + chr(0x4e25) +
    chr(0x683c) + chr(0x7684) + ' OCR ' + chr(0x7ea0) + chr(0x9519) + chr(0x4e13) +
    chr(0x5bb6) + chr(0xff0c) + chr(0x4ee5) + chr(0x4e0b) + chr(0x662f) + chr(0x521d) +
    chr(0x6b65) + chr(0x7684) + chr(0x5355) + chr(0x884c) + chr(0x6587) + chr(0x672c) +
    chr(0x8bc6) + chr(0x522b) + chr(0x7ed3) + chr(0x679c) + chr(0xff1a) + '\\n"' + nl +
    '            + "' + chr(0x3010) + ' " + T_A + " ' + chr(0x3011) + '\\n\\n"' + nl +
    '            + hint_block' + nl +
    '            + "' + chr(0x8bf7) + chr(0x7ed3) + chr(0x5408) + chr(0x63d0) +
    chr(0x4f9b) + chr(0x7684) + chr(0x56fe) + chr(0x50cf) + chr(0xff0c) + chr(0x4fee) +
    chr(0x6b63) + chr(0x4e0a) + chr(0x8ff0) + chr(0x6587) + chr(0x672c) + chr(0x4e2d) +
    chr(0x7684) + chr(0x9519) + chr(0x522b) + chr(0x5b57) + chr(0x6216) + chr(0x6f0f) +
    chr(0x5b57) + chr(0x3002) + '\\n"' + nl +
    '            + "**' + chr(0x6700) + chr(0x9ad8) + chr(0x7ea6) + chr(0x675f) +
    chr(0x7ea2) + chr(0x7ebf) + chr(0xff1a) + '**\\n"' + nl +
    '            + "1. ' + chr(0x5c3d) + chr(0x53ef) + chr(0x80fd) + chr(0x4fdd) +
    chr(0x6301) + chr(0x539f) + chr(0x53e5) + chr(0x539f) + chr(0x8c8c) + chr(0xff0c) +
    chr(0x7edd) + chr(0x5bf9) + chr(0x7981) + chr(0x6b62) + chr(0x5bf9) + chr(0x53e5) +
    chr(0x5b50) + chr(0x8fdb) + chr(0x884c) + chr(0x6da6) + chr(0x8272) + chr(0x3001) +
    chr(0x6539) + chr(0x5199) + chr(0x6216) + chr(0x5927) + chr(0x5e45) + chr(0x5ea6) +
    chr(0x589e) + chr(0x5220) + chr(0x3002) + '\\n"' + nl +
    '            + "2. ' + chr(0x5982) + chr(0x679c) + chr(0x8ba4) + chr(0x4e3a) +
    chr(0x6ca1) + chr(0x6709) + chr(0x9519) + chr(0x8bef) + chr(0xff0c) + chr(0x8bf7) +
    chr(0x76f4) + chr(0x63a5) + chr(0x539f) + chr(0x6837) + chr(0x8f93) + chr(0x51fa) +
    chr(0x3002) + '\\n\\n"' + nl +
    '            + "' + chr(0x8bf7) + chr(0x76f4) + chr(0x63a5) + chr(0x8f93) +
    chr(0x51fa) + chr(0x4fee) + chr(0x6b63) + chr(0x540e) + chr(0x7684) + chr(0x5b8c) +
    chr(0x6574) + chr(0x6587) + chr(0x672c) + chr(0xff0c) + chr(0x4e0d) + chr(0x8981) +
    chr(0x4efb) + chr(0x4f55) + chr(0x89e3) + chr(0x91ca) + chr(0x6216) + chr(0x591a) +
    chr(0x4f59) + chr(0x5b57) + chr(0x7b26) + chr(0xff1a) + '"' + nl +
    '        )' + nl +
    '        return {' + nl +
    '            "system_prompt": system_prompt,' + nl +
    '            "user_prompt": user_prompt,' + nl +
    '            "image_path": image_path,' + nl +
    '            "prompt_type": "targeted_correction",' + nl +
    '            "min_conf_idx": min_conf_idx,' + nl +
    '            "domain": domain,' + nl +
    '        }' + nl +
    nl
)

new_lines = [l + nl if not l.endswith(nl) else l for l in new_method.splitlines()]

lines[start:end] = new_lines
path.write_text(''.join(lines), encoding='utf-8')

src = path.read_text(encoding='utf-8')
try:
    ast.parse(src)
    print('Syntax OK')
except SyntaxError as e:
    ctx = src.splitlines()
    print(f'ERROR line {e.lineno}: {e.msg}')
    for i in range(max(0, e.lineno-2), min(len(ctx), e.lineno+1)):
        print(f'  {i+1}: {repr(ctx[i])}')
