"""Fix Jansen1995_extracted.yaml for basic LinkML loading.

Changes:
1. simulation_experiments: -> experiments: (match LinkML field name)
2. Remove keyed wrapper lines (JansenRit1995:, JansenRit1995_Delayed:)
3. Un-indent dynamics content by 2 spaces to flat format
"""

filepath = 'database/studies/Jansen1995/Jansen1995_extracted.yaml'
with open(filepath) as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # Rename the key
    if line.strip() == 'simulation_experiments:':
        new_lines.append(line.replace('simulation_experiments:', 'experiments:'))
        i += 1
        continue

    # Remove keyed wrapper lines and un-indent content below them
    if line.rstrip() in ('      JansenRit1995:', '      JansenRit1995_Delayed:'):
        # Skip this wrapper line
        i += 1
        # Un-indent everything below until we hit a line at 4 spaces indent or less
        while i < len(lines):
            next_line = lines[i]
            stripped = next_line.lstrip()
            if stripped == '' or stripped.startswith('#'):
                indent = len(next_line) - len(next_line.lstrip())
                if indent >= 8:
                    new_lines.append(next_line[2:])
                else:
                    new_lines.append(next_line)
                i += 1
                continue
            indent = len(next_line) - len(stripped)
            if indent <= 4:
                # Reached a sibling field (integration:, connectivity:, etc.)
                break
            if next_line[:8] == '        ':
                new_lines.append(next_line[2:])
            else:
                new_lines.append(next_line)
            i += 1
        continue

    new_lines.append(line)
    i += 1

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print(f'Done. {len(lines)} -> {len(new_lines)} lines')
