#!/bin/bash

PROJECT_NAME=$1

if [ -z "$PROJECT_NAME" ]; then
  echo "Usage: $0 <project_name>"
  exit 1
fi

mkdir -p "$PROJECT_NAME"/{.vscode,src/$PROJECT_NAME,tests}

cd "$PROJECT_NAME" || exit

# 仮想環境
python3 -m venv .venv

# __init__.py と main.py
touch src/$PROJECT_NAME/__init__.py
cat << EOF > src/$PROJECT_NAME/main.py
def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
EOF

# テストファイル
cat << EOF > tests/test_dummy.py
def test_dummy():
    assert True
EOF

# .gitignore（充実版）
cat << EOF > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
.venv/
env/
venv/
ENV/

# VSCode settings
.vscode/

# dotenv
.env
.env.*

# MacOS
.DS_Store

# Test & coverage
.coverage
coverage.*
htmlcov/
.tox/
nosetests.xml
pytest_cache/
.cache/

# Ruff cache
ruff_cache/

# MyPy cache
.mypy_cache/

# PyInstaller
build/
dist/
*.spec
EOF

# requirements.txt（空）
touch requirements.txt

# pyproject.toml（Ruff設定）
cat << EOF > pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "B"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.ruff.lint.isort]
force-single-line = true
EOF

# VSCode設定
cat << EOF > .vscode/settings.json
{
  "python.defaultInterpreterPath": "\${workspaceFolder}/.venv/bin/python",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll": true
  },
  "python.linting.enabled": false, // Ruff uses its own linting
  "ruff.enable": true,
  "ruff.formatOnSave": true
}
EOF

echo "✅ Python project '$PROJECT_NAME' has been created."

echo "🔧 次の手順:"
echo "1. cd $PROJECT_NAME"
echo "2. source .venv/bin/activate"
echo "3. pip install ruff"
echo "4. python src/$PROJECT_NAME/main.py"
