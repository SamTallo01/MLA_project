# MLA_project

## Project setup

On Windows, follow these steps to create a virtual environment, install dependencies, and manage the `requirements.txt` file:

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# 2. Install the packages you need
pip install torch torchvision torchaudio
pip install pandas matplotlib

# 3. Save installed dependencies into requirements.txt
pip freeze > requirements.txt

# 4. Reinstall everything later or on another machine
pip install -r requirements.txt

# 5. Every time you add or remove packages, update requirements.txt
pip freeze > requirements.txt
