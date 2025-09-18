import json, pathlib
md_path = pathlib.Path('llm/transformer/手撕Transformer.md')
out_path = pathlib.Path('llm/transformer/transformer.ipynb')
text = md_path.read_text(encoding='utf-8')
lines = text.splitlines(keepends=True)

cells = []
md_buffer = []
collect_code = False
code_buffer = []
code_lang = None
for line in lines:
    if not collect_code:
        if line.lstrip().startswith('```python'):
            # flush markdown buffer
            if md_buffer:
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {'language': 'markdown'},
                    'source': md_buffer
                })
                md_buffer = []
            collect_code = True
            code_lang = 'python'
            code_buffer = []
            continue  # skip fence line
        else:
            md_buffer.append(line)
    else:
        # collecting code
        if line.lstrip().startswith('```'):
            # end code block
            cells.append({
                'cell_type': 'code',
                'metadata': {'language': 'python'},
                'execution_count': None,
                'outputs': [],
                'source': code_buffer
            })
            collect_code = False
            code_buffer = []
            code_lang = None
        else:
            code_buffer.append(line)
# flush remaining markdown
if code_buffer:  # unclosed, treat as code anyway
    cells.append({
        'cell_type': 'code',
        'metadata': {'language': 'python'},
        'execution_count': None,
        'outputs': [],
        'source': code_buffer
    })
if md_buffer:
    cells.append({
        'cell_type': 'markdown',
        'metadata': {'language': 'markdown'},
        'source': md_buffer
    })

nb = {
    'cells': cells,
    'metadata': {
        'language_info': {'name': 'python'},
        'orig_nbformat': 4
    },
    'nbformat': 4,
    'nbformat_minor': 5
}
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {len(cells)} cells to', out_path)
