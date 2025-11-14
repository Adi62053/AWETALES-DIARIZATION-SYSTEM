# fix_config_path.py
with open('src/utils/config.py', 'r', encoding='utf-8') as f:
    content = f.read()

if '../configs' in content:
    fixed_content = content.replace('../configs', 'configs')
    with open('src/utils/config.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print(' Fixed config paths in src/utils/config.py')
else:
    print(' Config paths are already correct')
