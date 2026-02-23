import sys
from pathlib import Path
from io import StringIO
import traceback

def process_file(filepath: Path) -> int:
    errors = 0
    try:
        content = filepath.read_text('utf-8')
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return errors

    lines = content.splitlines()
    new_lines = []
    
    in_block = False
    in_output = False
    globals_dict = {}
    
    code_block = []
    last_output = ""
    
    i = 0
    changed = False
    
    while i < len(lines):
        line = lines[i]
        
        if in_output:
            if line.strip() == '<!-- /output -->':
                in_output = False
                new_lines.append(line)
            i += 1
            continue
            
        if not in_block:
            if line.startswith('```python'):
                in_block = True
                code_block = []
                new_lines.append(line)
            elif line.strip() == '<!-- output -->':
                in_output = True
                new_lines.append(line)
                if last_output:
                    new_lines.extend(last_output.splitlines())
                    changed = True
            else:
                new_lines.append(line)
            i += 1
        else:
            if line.startswith('```'):
                in_block = False
                new_lines.append(line)
                
                code_str = '\n'.join(code_block)
                
                # Look ahead to see if the next non-empty line is <!-- output -->
                is_tested = '# test: true' in code_str
                next_idx = i + 1
                while next_idx < len(lines):
                    if lines[next_idx].strip() == '':
                        next_idx += 1
                    elif lines[next_idx].strip() == '<!-- output -->':
                        is_tested = True
                        break
                    else:
                        break

                if is_tested and '# test: skip' not in code_str:
                    old_stdout = sys.stdout
                    redirected_output = sys.stdout = StringIO()
                    try:
                        exec(code_str, globals_dict)
                        out = redirected_output.getvalue().strip()
                        if out:
                            last_output = '\n' + out + '\n'
                        else:
                            last_output = ""
                    except Exception as e:
                        sys.stdout = old_stdout
                        print(f"Error executing code block in {filepath.name}:")
                        print(code_str)
                        traceback.print_exc()
                        errors += 1
                    finally:
                        sys.stdout = old_stdout
            else:
                code_block.append(line)
                new_lines.append(line)
            i += 1
            
    if changed:
        print(f"Updated {filepath}")
        filepath.write_text('\n'.join(new_lines) + '\n', 'utf-8')
        
    return errors

def main():
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("docs directory not found. Run from repo root.")
        sys.exit(1)
        
    total_errors = 0
    for filepath in docs_dir.rglob("*.md"):
        total_errors += process_file(filepath)
    for filepath in docs_dir.rglob("*.mdx"):
        total_errors += process_file(filepath)
        
    if total_errors > 0:
        print(f"Finished with {total_errors} errors.")
        sys.exit(1)
    else:
        print("All snippets evaluated successfully.")

if __name__ == '__main__':
    main()
