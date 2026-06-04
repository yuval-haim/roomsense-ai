# Git push guide

Use these commands to create a clean, logical multi-commit history from the project files.

This does not fake dates or authors. It simply separates the work into meaningful commits, which is the normal way to present a project history.

## 1. Start from a fresh repository

```bash
cd roomsense-ai

git init
git branch -M main
git status
```

## 2. Commit the project skeleton

```bash
git add README.md LICENSE .gitignore pyproject.toml requirements.txt Dockerfile docker-compose.yml
git add data/README.md configs/
git commit -m "Initialize RoomSense AI project structure"
```

## 3. Commit the core pipeline

```bash
git add src/reasoning src/utils src/visualization src/models/depth_estimator.py src/models/vlm_reasoner.py src/pipeline.py
git commit -m "Add room analysis pipeline and spatial risk engine"
```

## 4. Commit the demo dataset and generated examples

```bash
git add data/demo_images data/annotations outputs/examples outputs/reports
git commit -m "Add real living room demo data and example outputs"
```

## 5. Commit the Grounding DINO detector

```bash
git add src/models/detector.py scripts/run_inference.py scripts/run_grounding_dino.py requirements-grounding.txt
git commit -m "Add Grounding DINO open-vocabulary detector backend"
```

## 6. Commit the API and Streamlit app

```bash
git add api app
git commit -m "Add FastAPI service and Streamlit demo app"
```

## 7. Commit the evaluation scripts

```bash
git add eval/
git commit -m "Add evaluation scripts for detection risk grounding and latency"
```

## 8. Commit documentation updates

```bash
git add README.md docs/GIT_PUSH_GUIDE.md
git commit -m "Document Grounding DINO setup and project workflow"
```

## 9. Connect to GitHub and push

Create an empty GitHub repository named `roomsense-ai`, then run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/roomsense-ai.git
git push -u origin main
```

If the GitHub repository already has a README or commits, use this safer flow:

```bash
git remote add origin https://github.com/YOUR_USERNAME/roomsense-ai.git
git pull origin main --allow-unrelated-histories --rebase
git push -u origin main
```

If Git asks you to resolve conflicts, fix the files, then run:

```bash
git add .
git rebase --continue
git push -u origin main
```
